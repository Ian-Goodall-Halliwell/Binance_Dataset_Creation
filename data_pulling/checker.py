import csv
import os
from datetime import datetime, timedelta, timezone
import dateparser
import pytz
import pandas as pd

# type = 'day'
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


def import_csv(csvfilename: str) -> list:
    """
    Import a CSV file and return a list of lists.

    :param csvfilename: The name of the CSV file to import.
    :return: A list of lists.
    """
    data = []
    row_index = 0
    data = pd.read_feather(csvfilename).values.tolist()
    return data


def rounded_to_the_last_epoch(now: str) -> datetime:
    """
    Returns the datetime rounded to the last epoch.

    Args:
        now: The datetime to be rounded.

    Returns:
        The datetime rounded to the last epoch.
    """
    now = dateparser.parse(now)
    rounded = now - (now - datetime.min) % timedelta(minutes=360)
    return rounded


def rounded_to_the_last_epoch_1h(now):
    """
    This function rounds the time to the last hour.
    :param now: time to round
    :return: rounded time
    """
    now = dateparser.parse(now)
    rounded = now - (now - datetime.min) % timedelta(minutes=60)
    return rounded


def rounded_to_the_last_epoch_15m(now: str) -> datetime:
    """
    Returns the datetime rounded to the last 15 minutes.

    Args:
        now: The datetime to be rounded.

    Returns:
        The datetime rounded to the last 15 minutes.
    """
    now = dateparser.parse(now)
    rounded = now - (now - datetime.min) % timedelta(minutes=15)
    return rounded


def rounded_to_the_last_epoch_5m(now):
    """
    This function rounds the given time to the last 5 minutes.

    Args:
        now (str): The time to be rounded.

    Returns:
        datetime: The rounded time.
    """
    now = dateparser.parse(now)
    rounded = now - (now - datetime.min) % timedelta(minutes=5)
    return rounded


def rounded_to_the_last_epoch_1m(now: str) -> datetime:
    now = dateparser.parse(now)
    rounded = now - (now - datetime.min) % timedelta(minutes=1)
    return rounded


def degunk(type, end):
    if type == "6h":
        endv = rounded_to_the_last_epoch(end)
        end = endv.strftime("%Y-%m-%d %H:%M:%S")
    if type == "1h":
        endv = rounded_to_the_last_epoch_1h(end)
        end = endv.strftime("%Y-%m-%d %H:%M:%S")
    if type == "15m":
        endv = rounded_to_the_last_epoch_15m(end)
        end = endv.strftime("%Y-%m-%d %H:%M:%S")
    if type == "5m":
        endv = rounded_to_the_last_epoch_5m(end)
        end = endv.strftime("%Y-%m-%d %H:%M:%S")
    if type == "1m":
        endv = rounded_to_the_last_epoch_1m(end)
        end = endv.strftime("%Y-%m-%d %H:%M:%S")
    else:
        endiv = date_to_milliseconds(endv)
        end = datetime.fromtimestamp(endiv / 1000.0, tz=pytz.utc).strftime("%Y-%m-%d")
    path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data-download/{}-unprocessed".format(type),
    )
    if type == "1min":
        end = end + " 23:59:00"
        for a in os.listdir(path):
            if a == "CCI.csv":
                continue
            if not "BEAR" in a:
                if not "BULL" in a:
                    if not "UP" in a:
                        if not "DOWN" in a:
                            data = import_csv(os.path.join(path, a))
                            if data == []:
                                print(a, ": Null")
                                os.remove(os.path.join(path, a))
                                continue
                            last_row = data[-1]
                            if last_row[1] != end:
                                print(a, ":", last_row[1])
                                os.remove(os.path.join(path, a))
                        else:
                            os.remove(os.path.join(path, a))
                    else:
                        os.remove(os.path.join(path, a))
                else:
                    os.remove(os.path.join(path, a))
            else:
                os.remove(os.path.join(path, a))
    if type == "day":
        for a in os.listdir(path):
            if a == "CCI.csv":
                continue
            if not "BEAR" in a:
                if not "BULL" in a:
                    if not "UP" in a:
                        if not "DOWN" in a:
                            data = import_csv(os.path.join(path, a))
                            if data == []:
                                print(a, ": Null")
                                os.remove(os.path.join(path, a))
                                continue
                            last_row = data[-1]
                            if last_row[1] != end:
                                print(a, ":", last_row[1])
                                os.remove(os.path.join(path, a))
                        else:
                            os.remove(os.path.join(path, a))
                    else:
                        os.remove(os.path.join(path, a))
                else:
                    os.remove(os.path.join(path, a))
            else:
                os.remove(os.path.join(path, a))
    if type == "5min":
        end = end + " 23:55:00"
        for a in os.listdir(path):
            if a == "CCI.csv":
                continue
            if not "BEAR" in a:
                if not "BULL" in a:
                    if not "UP" in a:
                        if not "DOWN" in a:
                            data = import_csv(os.path.join(path, a))
                            if data == []:
                                print(a, ": Null")
                                os.remove(os.path.join(path, a))
                                continue
                            last_row = data[-1]
                            if last_row[1] != end:
                                print(a, ":", last_row[1])
                                os.remove(os.path.join(path, a))
                        else:
                            os.remove(os.path.join(path, a))
                    else:
                        os.remove(os.path.join(path, a))
                else:
                    os.remove(os.path.join(path, a))
            else:
                os.remove(os.path.join(path, a))
    if type == "30min":
        end = end + " 23:30:00"
        for a in os.listdir(path):
            if a == "CCI.csv":
                continue
            if not "BEAR" in a:
                if not "BULL" in a:
                    if not "UP" in a:
                        if not "DOWN" in a:
                            data = import_csv(os.path.join(path, a))
                            if data == []:
                                print(a, ": Null")
                                os.remove(os.path.join(path, a))
                                continue
                            last_row = data[-1]
                            if last_row[1] != end:
                                print(a, ":", last_row[1])
                                os.remove(os.path.join(path, a))
                        else:
                            os.remove(os.path.join(path, a))
                    else:
                        os.remove(os.path.join(path, a))
                else:
                    os.remove(os.path.join(path, a))
            else:
                os.remove(os.path.join(path, a))
    if type == "6h":
        for a in os.listdir(path):
            if a == "CCI.csv":
                continue
            if not "BEAR" in a:
                if not "BULL" in a:
                    if not "UP" in a:
                        if not "DOWN" in a:
                            data = import_csv(os.path.join(path, a))
                            if data == []:
                                print(a, ": Null")
                                os.remove(os.path.join(path, a))
                                continue
                            last_row = data[-1]
                            if last_row[1] != end:
                                print(a, ":", last_row[1])
                                os.remove(os.path.join(path, a))
                        else:
                            os.remove(os.path.join(path, a))
                    else:
                        os.remove(os.path.join(path, a))
                else:
                    os.remove(os.path.join(path, a))
            else:
                os.remove(os.path.join(path, a))
    if type == "1h":
        for a in os.listdir(path):
            if a == "CCI.csv":
                continue
            if not "BEAR" in a:
                if not "BULL" in a:
                    if not "UP" in a:
                        if not "DOWN" in a:
                            data = import_csv(os.path.join(path, a))
                            if data == []:
                                print(a, ": Null")
                                os.remove(os.path.join(path, a))
                                continue
                            last_row = data[-1]
                            if last_row[1] != end:
                                print(a, ":", last_row[1])
                                os.remove(os.path.join(path, a))
                        else:
                            os.remove(os.path.join(path, a))
                    else:
                        os.remove(os.path.join(path, a))
                else:
                    os.remove(os.path.join(path, a))
            else:
                os.remove(os.path.join(path, a))
    if type == "15m":
        for a in os.listdir(path):
            if a == "CCI.csv":
                continue
            if not "BEAR" in a:
                if not "BULL" in a:
                    if not "UP" in a:
                        if not "DOWN" in a:
                            data = import_csv(os.path.join(path, a))
                            if data == []:
                                print(a, ": Null")
                                os.remove(os.path.join(path, a))
                                continue
                            last_row = data[-1]
                            if last_row[1] != end:
                                print(a, ":", last_row[1])
                                os.remove(os.path.join(path, a))
                        else:
                            os.remove(os.path.join(path, a))
                    else:
                        os.remove(os.path.join(path, a))
                else:
                    os.remove(os.path.join(path, a))
            else:
                os.remove(os.path.join(path, a))
    if type == "5m":
        for a in os.listdir(path):
            if a == "CCI.csv":
                continue
            if not "BEAR" in a:
                if not "BULL" in a:
                    if not "UP" in a:
                        if not "DOWN" in a:
                            data = import_csv(os.path.join(path, a))
                            if data == []:
                                print(a, ": Null")
                                # os.remove(os.path.join(path,a))
                                continue
                            last_row = data[-1]
                            if last_row[1].rsplit("-", 1)[0] != end.rsplit("-", 1)[0]:
                                print(a, ":", last_row[1])
                                os.remove(os.path.join(path, a))
                        else:
                            os.remove(os.path.join(path, a))
                    else:
                        os.remove(os.path.join(path, a))
                else:
                    os.remove(os.path.join(path, a))
            else:
                os.remove(os.path.join(path, a))


def degunkapp(type, end):
    if type == "6h":
        endv = rounded_to_the_last_epoch(end)
        end = endv.strftime("%Y-%m-%d %H:%M:%S")
    if type == "1h":
        endv = rounded_to_the_last_epoch_1h(end)
        end = endv.strftime("%Y-%m-%d %H:%M:%S")
    if type == "15m":
        endv = rounded_to_the_last_epoch_15m(end)
        end = endv.strftime("%Y-%m-%d %H:%M:%S")
    if type == "5m":
        endv = rounded_to_the_last_epoch_5m(end)
        end = endv.strftime("%Y-%m-%d %H:%M:%S")
    else:
        endiv = date_to_milliseconds(endv)
        end = datetime.fromtimestamp(endiv / 1000.0, tz=pytz.utc).strftime("%Y-%m-%d")
    path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data-download/{}-temp".format(type),
    )
    if type == "1min":
        end = end + " 23:59:00"
        for a in os.listdir(path):
            if a == "CCI.csv":
                continue
            if not "BEAR" in a:
                if not "BULL" in a:
                    if not "UP" in a:
                        if not "DOWN" in a:
                            data = import_csv(os.path.join(path, a))
                            if data == []:
                                print(a, ": Null")
                                os.remove(os.path.join(path, a))
                                continue
                            last_row = data[-1]
                            if last_row[1] != end:
                                print(a, ":", last_row[1])
                                os.remove(os.path.join(path, a))
                        else:
                            os.remove(os.path.join(path, a))
                    else:
                        os.remove(os.path.join(path, a))
                else:
                    os.remove(os.path.join(path, a))
            else:
                os.remove(os.path.join(path, a))
    if type == "day":
        for a in os.listdir(path):
            if a == "CCI.csv":
                continue
            if not "BEAR" in a:
                if not "BULL" in a:
                    if not "UP" in a:
                        if not "DOWN" in a:
                            data = import_csv(os.path.join(path, a))
                            if data == []:
                                print(a, ": Null")
                                os.remove(os.path.join(path, a))
                                continue
                            last_row = data[-1]
                            if last_row[1] != end:
                                print(a, ":", last_row[1])
                                os.remove(os.path.join(path, a))
                        else:
                            os.remove(os.path.join(path, a))
                    else:
                        os.remove(os.path.join(path, a))
                else:
                    os.remove(os.path.join(path, a))
            else:
                os.remove(os.path.join(path, a))
    if type == "5min":
        end = end + " 23:55:00"
        for a in os.listdir(path):
            if a == "CCI.csv":
                continue
            if not "BEAR" in a:
                if not "BULL" in a:
                    if not "UP" in a:
                        if not "DOWN" in a:
                            data = import_csv(os.path.join(path, a))
                            if data == []:
                                print(a, ": Null")
                                os.remove(os.path.join(path, a))
                                continue
                            last_row = data[-1]
                            if last_row[1] != end:
                                print(a, ":", last_row[1])
                                os.remove(os.path.join(path, a))
                        else:
                            os.remove(os.path.join(path, a))
                    else:
                        os.remove(os.path.join(path, a))
                else:
                    os.remove(os.path.join(path, a))
            else:
                os.remove(os.path.join(path, a))
    if type == "30min":
        end = end + " 23:30:00"
        for a in os.listdir(path):
            if a == "CCI.csv":
                continue
            if not "BEAR" in a:
                if not "BULL" in a:
                    if not "UP" in a:
                        if not "DOWN" in a:
                            data = import_csv(os.path.join(path, a))
                            if data == []:
                                print(a, ": Null")
                                os.remove(os.path.join(path, a))
                                continue
                            last_row = data[-1]
                            if last_row[1] != end:
                                print(a, ":", last_row[1])
                                os.remove(os.path.join(path, a))
                        else:
                            os.remove(os.path.join(path, a))
                    else:
                        os.remove(os.path.join(path, a))
                else:
                    os.remove(os.path.join(path, a))
            else:
                os.remove(os.path.join(path, a))
    if type == "6h":
        for a in os.listdir(path):
            if a == "CCI.csv":
                continue
            if not "BEAR" in a:
                if not "BULL" in a:
                    if not "UP" in a:
                        if not "DOWN" in a:
                            data = import_csv(os.path.join(path, a))
                            if data == []:
                                print(a, ": Null")
                                os.remove(os.path.join(path, a))
                                continue
                            last_row = data[-1]
                            if last_row[1] != end:
                                print(a, ":", last_row[1])
                                os.remove(os.path.join(path, a))
                        else:
                            os.remove(os.path.join(path, a))
                    else:
                        os.remove(os.path.join(path, a))
                else:
                    os.remove(os.path.join(path, a))
            else:
                os.remove(os.path.join(path, a))
    if type == "1h":
        for a in os.listdir(path):
            if a == "CCI.csv":
                continue
            if not "BEAR" in a:
                if not "BULL" in a:
                    if not "UP" in a:
                        if not "DOWN" in a:
                            data = import_csv(os.path.join(path, a))
                            if data == []:
                                print(a, ": Null")
                                os.remove(os.path.join(path, a))
                                continue
                            last_row = data[-1]
                            if last_row[1] != end:
                                print(a, ":", last_row[1])
                                os.remove(os.path.join(path, a))
                        else:
                            os.remove(os.path.join(path, a))
                    else:
                        os.remove(os.path.join(path, a))
                else:
                    os.remove(os.path.join(path, a))
            else:
                os.remove(os.path.join(path, a))
    if type == "15m":
        for a in os.listdir(path):
            if a == "CCI.csv":
                continue
            if not "BEAR" in a:
                if not "BULL" in a:
                    if not "UP" in a:
                        if not "DOWN" in a:
                            data = import_csv(os.path.join(path, a))
                            if data == []:
                                print(a, ": Null")
                                os.remove(os.path.join(path, a))
                                continue
                            last_row = data[-1]
                            if last_row[1] != end:
                                print(a, ":", last_row[1])
                                # os.remove(os.path.join(path,a))
                        else:
                            os.remove(os.path.join(path, a))
                    else:
                        os.remove(os.path.join(path, a))
                else:
                    os.remove(os.path.join(path, a))
            else:
                os.remove(os.path.join(path, a))
    if type == "5m":
        for a in os.listdir(path):
            if a == "CCI.csv":
                continue
            if not "BEAR" in a:
                if not "BULL" in a:
                    if not "UP" in a:
                        if not "DOWN" in a:
                            data = import_csv(os.path.join(path, a))
                            if data == []:
                                print(a, ": Null")
                                # os.remove(os.path.join(path,a))
                                continue
                            last_row = data[-1]
                            if last_row[1] != end:
                                print(a, ":", last_row[1])
                                # os.remove(os.path.join(path,a))
                        else:
                            os.remove(os.path.join(path, a))
                    else:
                        os.remove(os.path.join(path, a))
                else:
                    os.remove(os.path.join(path, a))
            else:
                os.remove(os.path.join(path, a))


def makesame():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data-download")
    # path1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data-download/day-unprocessed')
    alist = []
    for a in os.listdir(path + "/day-unprocessed"):
        alist.append(a)
    for bbd in [
        path + "/5min-unprocessed",
        path + "/30min-unprocessed",
        path + "/1min-unprocessed",
    ]:
        blist = []
        for b in os.listdir(bbd):
            blist.append(b)
        for elem in alist:
            if not elem in blist:
                print("a", elem)
        for elmt in blist:
            if not elmt in alist:
                print("b", elmt)
                # os.remove(os.path.join(bbd,elmt))


if __name__ == "__main__":
    degunk("5m", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
