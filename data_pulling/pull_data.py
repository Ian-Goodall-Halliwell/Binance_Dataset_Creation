import csv
import os
import queue
import shutil
import time
from datetime import datetime, timedelta
from threading import Lock
import dateparser

import pytz
from binance.client import Client
from binance.enums import HistoricalKlinesType
from joblib import Parallel, delayed
from typing import List
import pandas as pd
from ta.volume import volume_weighted_average_price
import numpy as np

# import dill as pickle

try:
    import checker
except:
    from data_pulling import checker
s_print_lock = Lock()
try:
    from data_pulling import dump_bin
except:
    import dump_bin


def call_api(func_to_call, client, clientlist, params, wait=False):

    x = getattr(client, func_to_call)(**params)
    out = []
    while True:
        try:
            z = next(x)
            out.append(z)
        except StopIteration:
            break
        except Exception as e:
            print(e)
            if wait:
                time.sleep(10)

            if out != []:
                try:
                    params["start_str"] = out[-1]["T"]
                except:
                    params["start_str"] = out[-1][0]
            client = clientlist.get()

            client, v = call_api(func_to_call, client, clientlist, params, wait=True)
            out.extend(v)
            break
    x = out

    return client, x


def date_to_milliseconds(date_str: str) -> int:
    """Convert UTC date to milliseconds
    If using offset strings add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"
    See dateparse docs for formats http://dateparser.readthedocs.io/en/latest/
    :param date_str: date in readable format, i.e. "January 01, 2018", "11 hours ago UTC", "now UTC"
    :type date_str: str
    """
    # get epoch value in UTC
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    # parse our date string
    d = dateparser.parse(date_str)
    # if the date is not timezone aware apply UTC timezone
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)
    # return the difference in time
    return int((d - epoch).total_seconds() * 1000.0)


def get_exchange_info(client) -> dict:
    """
    Get exchange info
    :return: dict
    """
    # time.sleep(1)
    
    return client.get_exchange_info()


def getstate(
    outlist: List[str], exchangeinfo: dict,client, tokennames: List[str] = ["BUSD"]
) -> List[str]:
    """
    This function returns a list of tokens.

    Args:
        outlist: A list of tokens.
        exchangeinfo: A dictionary of exchange information.
        tokennames: A list of token names.

    Returns:
        A list of tokens.
    """
    
    
    currlist = []
    for ab in exchangeinfo["symbols"]:
        if (
            ab["quoteAsset"] == tokennames[0]
            and ab["symbol"] not in outlist
            and "BEAR" not in ab["symbol"]
            and "BULL" not in ab["symbol"]
            and "UP" not in ab["symbol"]
            and "DOWN" not in ab["symbol"]
        ):
            sm = client.get_klines(
                symbol=ab["symbol"],
                interval=Client.KLINE_INTERVAL_1DAY,
                limit=10,
            )
            
            sm.pop(-1)
            vol = [float(x[5]) for x in sm]
            v = [float(x[4]) for x in sm]
            vol = sum(vol) / len(vol)
            v = sum(v) / len(v)
            
            vv = vol * v
            if vv > 1000000:
                currlist.append(ab["symbol"])
        if "DEFI" in ab["symbol"]:
            currlist.append(ab["symbol"])
            # if ab['quoteAsset'] == tokennames[1]:
            #     if not ab['symbol'] in outlist:
            #         if not "BEAR" in ab['symbol']:
            #             if not "BULL" in ab['symbol']:
            #                 currlist.append(ab['symbol'])
    downloaded = os.listdir("F:/binancedata/1m-raw")
    downloaded = [x.split(".")[0] for x in downloaded]
    currlist = list(set(currlist).difference(downloaded))
    print("number of tokens:", len(currlist))
    return currlist


def download1(
    start: str, end: str, interval: str, q, path: str, type1: str, token, trade=False
):
    """
    This function downloads the data from the binance API
    :param start: start date
    :param end: end date
    :param interval: interval of the data
    :param q: queue
    :param pth: path to save the data
    :param type1: type of data
    :param client: client
    :return:
    """

    client = q.get()
    # starts = start // 1000  # ,
    # ends = end // 1000  # , tz=pytz.utc
    # .strftime("%Y-%m-%d %H:%M:%S")

    client, klines = call_api(
        client=client,
        func_to_call="get_historical_klines_generator",
        clientlist=q,
        params=dict(
            symbol=token,
            interval=interval,
            start_str=start,
            end_str=end,
            klines_type=HistoricalKlinesType.SPOT,
        ),
    )
    # klines = client.get_historical_klines(
    #     token,
    #     interval,
    #     start_str=start,
    #     end_str=end,
    #     klines_type=HistoricalKlinesType.SPOT,
    # )
    # strttme = int(dateparser.parse(start).timestamp() * 1000)
    # strd = dateparser.parse(start).astimezone(pytz.utc).timestamp() * 1000
    if trade == True:

        client, trades = call_api(
            client=client,
            func_to_call="aggregate_trade_iter",
            clientlist=q,
            params=dict(
                symbol=token,
                start_str=start,
            ),
        )
        if trades == []:
            return
        trades_ = pd.DataFrame(trades)
        idx = trades_["T"].apply(
            lambda x: datetime.fromtimestamp(x / 1000, tz=pytz.utc)
            .replace(microsecond=0, second=0)
            .strftime("%Y-%m-%d %H:%M:%S")
        )
        dft = [
            [
                int(x["T"]),
                np.float32(x["p"]),
                np.float32(x["q"]),
                bool(x["m"]),
                bool(x["M"]),
            ]
            for x in trades
        ]
        dts = [list(idx.values), dft]
        df = pd.DataFrame(dts, index=["date", "trades"], columns=idx).T
        df = df.sort_values(by="date")
        df = df.groupby(by=["date"])["trades"].apply(lambda x: x.values)
    if klines == []:
        # q.put(client)
        return

    headers = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "symbol",
        "VWAP",
    ]
    klineframe = pd.DataFrame(columns=headers)
    klineframe["date"] = [
        datetime.fromtimestamp(x[0] / 1000.0, tz=pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
        for x in klines
    ]
    lastdate = klineframe["date"].iloc[-1]
    klineframe["open"] = [x[1] for x in klines]
    klineframe["high"] = [x[2] for x in klines]
    klineframe["low"] = [x[3] for x in klines]
    klineframe["close"] = [x[4] for x in klines]
    klineframe["volume"] = [x[5] for x in klines]
    klineframe["symbol"] = [token for _ in klines]
    if trade:
        df = df.apply(lambda x: str(x))

        klineframe = klineframe.join(df, on="date")

    klineframe = klineframe.astype(
        {
            "date": "datetime64[ns]",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
            "symbol": "object",
            "VWAP": "float64",
        }
    )
    klineframe["VWAP"] = volume_weighted_average_price(
        klineframe["high"],
        klineframe["low"],
        klineframe["close"],
        klineframe["volume"],
        window=1444,
        fillna=True,
    )
    klineframe.to_feather(f"{path}/{token}.feather")
    # store = pd.HDFStore(f"{path}/{token}.h5")
    # store['df'] = klineframe
    # klineframe.to_csv(f"{path}/{token}.csv")
    # q.put(client)


def startdownload_1m(start, end, dir, app=False, clients=[], trades=False):
    """
    This is a multi-line Google style docstring.

    Args:
        start (str): Start date in format YYYY-MM-DD
        end (str): End date in format YYYY-MM-DD
        dir (str): Directory to save the data
        app (bool): Whether to append to existing data
        clients (list): List of clients to use

    Returns:
        None
    """
    if not os.path.exists(dir):
        os.mkdir(dir)
    exchg = get_exchange_info(client=clients[0])
    if app == True:
        currlist = [
            popl.split(".")[0]
            for popl in os.listdir(
                "C:/Users/Ian/Documents/Financial testing/data_pulling/data-download/1m-raw"
            )
            if popl.split(".")[0] != "CCI"
        ]
    else:
        currlist = getstate([], exchg,client=clients[0])
    fuln = len(currlist) // len(clients)
    q = queue.SimpleQueue()
    for _ in range(fuln * 10000):
        for item in clients:
            q.put(item)
    interval = Client.KLINE_INTERVAL_1MINUTE

    cs = len(clients) // 4
    Parallel(n_jobs=cs, backend="threading")(
        delayed(download1)(int(start), int(end), interval, q, dir, "1m", i, trades)
        for i in currlist
    )


def delete_incompletes(pth):
    def import_csv(csvfilename):
        data = []
        row_index = 0
        with open(csvfilename, "r", encoding="utf-8", errors="ignore") as scraped:
            reader = csv.reader(scraped, delimiter=",")
            for row in reader:
                if row:  # avoid blank lines
                    row_index += 1
                    columns = [str(row_index), row[0]]
                    data.append(columns)
        return data

    for a in os.listdir(pth):
        data = import_csv(os.path.join(pth, a))
        last_row = data[-1]
        if last_row[1] != "2022-01-04 23:59:00":
            os.remove(os.path.join(pth, a))


def interval_to_milliseconds(interval):
    """Convert a Binance interval string to milliseconds
    :param interval: Binance interval string 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w
    :type interval: str
    :return:
         None if unit not one of m, h, d or w
         None if string not in correct format
         int value of interval in milliseconds
    """
    ms = None
    seconds_per_unit = {"m": 60, "h": 60 * 60, "d": 24 * 60 * 60, "w": 7 * 24 * 60 * 60}
    unit = interval[-1]
    if unit in seconds_per_unit:
        try:
            ms = int(interval[:-1]) * seconds_per_unit[unit] * 1000
        except ValueError:
            pass
    return ms


def clean_assets(client):
    dustables = client.get_dust_assets()
    dust_list = ""
    for en, item in enumerate(dustables["details"]):
        if float(item["toBNB"]) == 0:
            continue
        if item["asset"] == "BUSD":
            continue
        if item["asset"] == "VTHO":
            continue
        if item["asset"] == "VTHOBUSD":
            continue
        if (en + 1) < len(dustables["details"]):
            dust_list = dust_list + item["asset"] + ","
        else:
            dust_list = dust_list + item["asset"]
    dst = client.transfer_dust(asset=dust_list)


def get_acct(client):
    snaps = client.get_account()
    return [ast for ast in snaps["balances"] if float(ast["free"]) != 0]


def run1m(d, clilist, trades=False):
    csv_path = "F:/binancedata/1m-raw"
    qlib_dir = "F:/binancedata/1m-qlib"
    if os.path.exists(csv_path):
        shutil.rmtree(csv_path)
    os.mkdir(csv_path)
    dmin = d.strftime("%Y-%m-%d %H:%M:%S")
    strt = dateparser.parse("2019-01-01 00:00:00")
    strt = "2019-01-01 00:00:00"
    strt = date_to_milliseconds(strt)
    dmin = date_to_milliseconds(dmin)
    startdownload_1m(start=strt, end=dmin, dir=csv_path, clients=clilist, trades=trades)
    time.sleep(30)
    # checker.degunk("1m", d.strftime("%d %B, %Y, %H:%M:%S"))
    if os.path.exists(qlib_dir):
        shutil.rmtree(qlib_dir)
    os.mkdir(qlib_dir)
    for clin in clilist:
        clin.close_connection()
    b = dump_bin.DumpDataAll(
        csv_path=csv_path,
        qlib_dir=qlib_dir,
        include_fields=f"""open,high,low,close,volume,VWAP{",trades" if trades == True else ''}""",
        freq="1min",
        file_suffix=".feather",
    )

    b.dump()


# @profile
def append1m(d, clilist):
    csv_path = "F:/binancedata/1m-temp"
    if os.path.exists(csv_path):
        shutil.rmtree(csv_path)
    os.mkdir(csv_path)
    strd = d.strftime("%d %B, %Y, %H:%M:%S")
    dmin = d + timedelta(minutes=60)
    dmax = d - timedelta(days=1)
    with open("F:/binancedata/1m-qlib/calendars/1min.txt") as f:
        read = csv.reader(f)
        cl = list(read)
        last = cl[-1][0]
    lastd = dateparser.parse(last)
    # lastd = lastd - timedelta(days=1)
    lastd = lastd - timedelta(minutes=5)
    startdownload_1m(
        start=lastd.strftime("%d %B, %Y, %H:%M:%S"),
        end=dmin.strftime("%d %B, %Y, %H:%M:%S"),
        dir=csv_path,
        app=True,
        clients=clilist,
    )
    time.sleep(1)
    # checker.degunkapp('5m',strd)
    # if os.path.exists(qlib_dir):
    #     shutil.rmtree(qlib_dir)
    # os.mkdir(qlib_dir)
    for clin in clilist:
        clin.close_connection()
    if os.listdir(csv_path) != []:
        qlib_dir = "F:/binancedata/1m-qlib/"
        b = dump_bin.DumpDataUpdate(
            csv_path=csv_path,
            qlib_dir=qlib_dir,
            include_fields="open,high,low,close,volume,VWAP,trades",
            freq="1min",
            max_workers=16,
        )
        b.dump()


if __name__ == "__main__":
    d = datetime.now(pytz.utc)
    with open("C:/Users/Ian/Desktop/clients.csv",'r') as f:
        reader = csv.reader(f)
        clients = [Client(row[0],row[1]) for row in reader]
    # clients = [


        
    # ]
    run1m(d, clilist=clients, trades=False)
    print(f"data download and prep time elapsed: {datetime.now(pytz.utc) - d}")
