
import os
import re

from multiprocessing import Manager
from typing import Dict, Type, Union,List
import dask.dataframe as dd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from ops import Operators, register_all_ops
from subutils import cache
import dateparser
from datetime import timedelta
import gc
from ta.volume import volume_weighted_average_price
from tqdm.asyncio import tqdm
def procfunc(lep, df, field: List[str], scores, symbol, substart, end_time,start_time):

    register_all_ops()
    print(field[1])
    out = lep.expression(df, field=field[0], start_time=substart, end_time=end_time)
    start_time = dateparser.parse(start_time) + timedelta(minutes=1)
    start_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
    out = out.rename(field[1])
    out = out.loc[start_time:]
    #out = dd.from_pandas(out,npartitions=1)
    #print(f"done field: {field[1]}")
    #gc.collect()
    return out


def priceadj(row):
    return (row["$close"] + row["$low"] + row["$high"]) / 3


def parse_field(field: str) -> str:
    # Following patterns will be matched:
    # - $close -> Feature("close")
    # - $close5 -> Feature("close5")
    # - $open+$close -> Feature("open")+Feature("close")
    # TODO: this maybe used in the feature if we want to support the computation of different frequency data
    # - $close@5min -> Feature("close", "5min")
    if not isinstance(field, str):
        field = str(field)
    # Chinese punctuation regex:
    # \u3001 -> 、
    # \uff1a -> ：
    # \uff08 -> (
    # \uff09 -> )
    chinese_punctuation_regex = r"\u3001\uff1a\uff08\uff09"
    for pattern, new in [
        (
            rf"\$\$([\w{chinese_punctuation_regex}]+)",
            r'PFeature("\1")',
        ),  # $$ must be before $
        (rf"\$([\w{chinese_punctuation_regex}]+)", r'Feature("\1")'),
        (r"(\w+\s*)\(", r"Operators.\1("),
    ]:  # Features  # Operators
        field = re.sub(pattern, new, field)
    return field


class ExpressionProvider:
    """Expression provider class
    Provide Expression data.
    """

    def __init__(self):
        self.expression_instance_cache = {}

    def get_expression_instance(self, field: str) -> str:
        try:
            if field in self.expression_instance_cache:
                expression = self.expression_instance_cache[field]
            else:
                field = parse_field(field)
                expression = eval(field)
                self.expression_instance_cache[field] = expression
        except Exception as e:
            print(e)
        return expression


def time_to_slc_point(t: Union[None, str, pd.Timestamp]) -> Union[None, pd.Timestamp]:  # type: ignore
    if t is None:
        # None represents unbounded in Qlib or Pandas(e.g. df.loc[slice(None, "20210303")]).
        return t
    else:
        return pd.Timestamp(t)


class LocalExpressionProvider(ExpressionProvider):
    """Local expression data provider class
    Provide expression data from local data source.
    """

    def __init__(self):
        super().__init__()

    def expression(
        self,
        instrument: str,
        field: str,
        start_time: str = None,
        end_time: str = None,
        freq: str = "1min",
    ):
        expression = self.get_expression_instance(field)
        start_time = time_to_slc_point(start_time)
        end_time = time_to_slc_point(end_time)
        start_index, end_index = query_start, query_end = start_time, end_time
        try:
            series = expression.load(instrument, query_start, query_end, freq)
        except Exception as e:
            raise e
        try:
            series = series.astype(np.float64)
        except ValueError:
            pass
        except TypeError:
            pass

        series = series.loc[start_index:end_index]
        return series
def fixvolume(df):
    df["$volume"] = df["$volume"].rolling(11,win_type="hamming",min_periods=1).mean()
    df["$volume"] = df["$volume"]*((df["$open"]+df["$close"]+df["$high"]+df["$low"])/4)
    return df
def procData(df, fieldlist, nproc, dates ,label=False,symbols=None):

    
    try:
        btcdata = cache("BTCBUSD.parquet")
        del btcdata

    except:
        btcdata = df[df["$symbol"] == "BTCBUSD"].compute()
        #btcdata = fixvolume(btcdata)
        cache("BTCBUSD", btcdata)
        del btcdata
    try:
        ethdata = cache("ETHBUSD.parquet")
        del ethdata

    except:
        ethdata = df[df["$symbol"] == "ETHBUSD"].compute()
        #ethdata = fixvolume(ethdata)
        cache("ETHBUSD", ethdata)
        del ethdata
    lep = LocalExpressionProvider()
    df = df.astype(
        {
            "$close": "float32",
            "$high": "float32",
            "$low": "float32",
            "$open": "float32",
            "$volume": "float32",
            "$vwap": "float32",
            "$adj": "float32",
        }
    )
    scores = {}
    pbar=  tqdm(desc="Processing data",total=(len(dates)*len(symbols)))
    en = 0
    for start_time, end_time in dates:
        substart = (dateparser.parse(start_time) - timedelta(days=1)).strftime("%Y-%m-%d")
        if nproc != 1:
            with Parallel(n_jobs=nproc,batch_size=10,timeout=999999,backend="loky") as parallel:
                for symbol in symbols:
                    subdf = df[df["$symbol"] == symbol].compute()
                    #subdf = fixvolume(subdf)
                    results = parallel(
                        delayed(procfunc)(
                            lep, subdf, field, scores, symbol, substart, end_time,start_time
                        )
                        for field in tqdm(fieldlist,desc=f"Calculating features for {symbol}", leave=False)
                    )
                    
                    del subdf
                    results = pd.concat(results,axis=1)
                    #results = dd.from_pandas(results, chunksize=40000)

                    if label == False:
                        path = cache(f"{symbol}_cached.parquet", results)
                    else:
                        path = cache(f"{symbol}_label_cached.parquet", results)
                    del results
                    #gc.collect()
                    scores.update({symbol: path})
                    pbar.update(1)
        else:
            for symbol in symbols:
                subdf = df[df["$symbol"] == symbol].compute()
                results = []
                for field in fieldlist:
                    results.append(
                        procfunc(lep, subdf, field, scores, symbol, substart, end_time)
                    )
                del subdf
                results = pd.DataFrame(results).T
                if label == False:
                    path = cache(f"{symbol}_cached.parquet", results)
                else:
                    path = cache(f"{symbol}_label_cached.parquet", results)
                scores[symbol] = path
                #gc.collect()
                del results
    return scores

def compiledata(path):
    try:
        os.remove(os.path.join(path, "dataset.h5"))
    except:
        pass
    for file in os.listdir(os.path.join(path, "1m-raw")):
        ds = pd.read_feather(os.path.join(os.path.join(path, "1m-raw"), file))
        ds = ds.rename(
            columns={
                "open": "$open",
                "high": "$high",
                "low": "$low",
                "close": "$close",
                "volume": "$volume",
                "symbol": "$symbol",
                "VWAP": "$vwap",
            }
        )
        ds = fixvolume(ds)
        ds['$VWAP'] = volume_weighted_average_price(high=ds['$high'],low=ds["$low"],close=ds["$close"],volume=ds["$volume"],window=1440)
       
        ds["$adj"] = ds.apply(priceadj, axis=1)
        ds = dd.from_pandas(ds, chunksize=100000)
        ds = ds.set_index("date")
        ds.astype(np.float32)
        ds.to_hdf(
            os.path.join(path, "dataset.h5"),
            key=file.split(".")[0],
            mode="a",
            format="table",
        )


if __name__ == "__main__":
    compiledata("data")
