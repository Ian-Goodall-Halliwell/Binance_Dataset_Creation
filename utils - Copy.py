
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
# def procfunc(lep, df, field: List[str], scores, symbol, substart, end_time,start_time):

    
#     return out

def procfuncold(lep, df, field: List[str], scores, symbol, substart, end_time,start_time):

    register_all_ops()
    #print(field[1])
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
def ppfc(df,symbol,lep,scores,substart,end_time,start_time,fieldlist,label):
    subdf = df[df["$symbol"] == symbol].compute()
    #subdf = fixvolume(subdf)
    import talib
    fs = talib.get_function_groups()
    results = []
    
    for field in tqdm(fieldlist):

        results.append(procfunc(
                lep, subdf, field, scores, symbol, substart, end_time,start_time
            ))
        
    
    
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

def ppfc_old(df,symbol,lep,scores,substart,end_time,start_time,fieldlist,label):
    subdf = df[df["$symbol"] == symbol].compute()
    #subdf = fixvolume(subdf)
    import talib
    fs = talib.get_function_groups()
    results = []
    
    for field in tqdm(fieldlist):

        results.append(procfunc(
                lep, subdf, field, scores, symbol, substart, end_time,start_time
            ))
        
    
    
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
def procData(df, fieldlist, nproc, dates ,label=False,symbols=None):
    trynew= False
    if trynew:
        rolling_windows = [
            3,
            5,
            6,
            9,
            12,
            15,
            30,
            60,
            120,
            180,
            240,
            360,
            480,
            720,
            1440,
            2880,
            3600,
            5760,
            10080,
            20160,
            40320
        ]
        from tuneta.tune_ta import TuneTA
        tt = TuneTA(n_jobs=10, verbose=True)
        X_train = df.compute().reset_index(drop=True)
        multi = ((pd.to_datetime(x),y) for x,y in zip(X_train["date"].values, X_train["$symbol"].values))
        multdex =  pd.MultiIndex.from_tuples(multi,names=["date","symbol"])
        X_train.index = multdex
        #X_train.index = pd.MultiIndex.from_tuples((pd.to_datetime(x),y) for x,y in zip(X_train["date"].values, X_train["$symbol"].values))
        X_train["target"] = X_train["$close"].diff(periods=-60) / X_train["$close"] * 100
        X_train = X_train.dropna(how="any")
        y_train = X_train["target"]
        X_train = X_train.drop(["target"],axis=1)
        X_train = X_train.drop(["$symbol","$vwap"],axis=1)
        X_train = X_train.rename(
            columns={
                "$open": "open",
                "$high": "high",
                "$low": "low",
                "$close": "close",
                "$volume": "volume",
            }
        )
        # import talib
        # fs = talib.get_function_groups()
        tt.fit(X_train, y_train,
            indicators=['tta'],
            ranges=rolling_windows,
            trials=100,
            early_stop=100,
            min_target_correlation=.02,
        )
        
        # Show time duration in seconds per indicator
        tt.fit_times()

        # Show correlation of indicators to target
        tt.report(target_corr=True, features_corr=True)

        # Select features with at most x correlation between each other
        tt.prune(max_inter_correlation=.7)

        # Show correlation of indicators to target and among themselves
        tt.report(target_corr=True, features_corr=True)

        # Add indicators to X_train
        features = tt.transform(X_train)
        X_train = pd.concat([X_train, features], axis=1)

        # Add same indicators to X_test
        features = tt.transform(X_test)
        X_test = pd.concat([X_test, features], axis=1)
    # try:
    #     btcdata = cache("BTCBUSD.parquet")
    #     del btcdata

    # except:
    #     btcdata = df[df["$symbol"] == "BTCBUSD"].compute()
    #     #btcdata = df.compute()
    #     #btcdata = fixvolume(btcdata)
    #     cache("BTCBUSD", btcdata)
    #     del btcdata
    # try:
    #     ethdata = cache("ETHBUSD.parquet")
    #     del ethdata

    # except:
    #     ethdata = df[df["$symbol"] == "ETHBUSD"].compute()
    #     #ethdata = fixvolume(ethdata)
    #     cache("ETHBUSD", ethdata)
    #     del ethdata
    lep = LocalExpressionProvider()
    # df = df.astype(
    #     {
    #         "$close": "float32",
    #         "$high": "float32",
    #         "$low": "float32",
    #         "$open": "float32",
    #         "$volume": "float32",
    #         "$vwap": "float32",
    #         "$adj": "float32",
    #     }
    # )
    scores = {}
    pbar=  tqdm(desc="Processing data",total=(len(dates)*len(symbols)))
    en = 0
    for start_time, end_time in dates:
        substart = (dateparser.parse(start_time) - timedelta(minutes=40330)).strftime("%Y-%m-%d")
        if nproc != 1:
            with Parallel(n_jobs=nproc,timeout=999999,backend="loky") as parallel:
                results = parallel(
                    delayed(ppfc)(
                        df,symbol,lep, scores, substart, end_time,start_time,fieldlist,label
                    )
                    for symbol in symbols
                )
                
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

def compiledata(path,append=True):
    if not append:
        try:
            os.remove(os.path.join(path, "dataset.h5"))
        except:
            pass
    
    for file in os.listdir(os.path.join(path, "1m-raw")):
        if append:
            
            df = pd.read_hdf(
                os.path.join(path, "dataset.h5"),
                key=file.split(".")[0],
                start=-1440
            )
            store = pd.HDFStore(os.path.join(path, "dataset.h5"))
            dfidx = df.index[-1440]
            endidx = df.index[-1]
            print('e')
        ds = pd.read_csv(os.path.join(os.path.join(path, "1m-raw"), file))
        ds = ds.set_index(pd.to_datetime(ds["date"]))
        if append:
            ds = ds[dfidx:]
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
        ds['$vwap'] = volume_weighted_average_price(high=ds['$high'],low=ds["$low"],close=ds["$close"],volume=ds["$volume"],window=1440)
       
        ds["$adj"] = ds.apply(priceadj, axis=1)
        #ds = dd.from_pandas(ds, chunksize=100000)
        
        ds.astype(
        {
            "date": "datetime64[ns]",
            "$open": "float32",
            "$high": "float32",
            "$low": "float32",
            "$close": "float32",
            "$volume": "float32",
            "$symbol": "object",
            "$vwap": "float32",
            "$adj": "float32",
        })
        if append:
            ds = ds[endidx + timedelta(minutes=1):]
            #ds = pd.concat([df,ds],axis=0)
            if not ds.empty:
                store.append(file.split(".")[0],ds)
            store.close()
        else:
            # print(ds.isna().sum())
            ds.to_hdf(
                os.path.join(path, "dataset.h5"),
                key=file.split(".")[0],
                mode="a",
                format="table",
            )


def procDataold(df, fieldlist, nproc, dates ,label=False,symbols=None):
    trynew= True
    if trynew:
        from tuneta.tune_ta import TuneTA
        tt = TuneTA(n_jobs=8, verbose=True)
        X_train = df.compute().reset_index(drop=True)
        multi = ((pd.to_datetime(x),y) for x,y in zip(X_train["date"].values, X_train["$symbol"].values))
        multdex =  pd.MultiIndex.from_tuples(multi,names=["date","symbol"])
        X_train.index = multdex
        X_train["target"] = X_train["$close"].diff(periods=-60) / X_train["$close"] * 100
        X_train = X_train.dropna(how="any")
        y_train = X_train["target"]
        X_train = X_train.drop(["target"],axis=1)
        X_train = X_train.drop(["$symbol","$vwap"],axis=1)
        X_train = X_train.rename(
            columns={
                "$open": "open",
                "$high": "high",
                "$low": "low",
                "$close": "close",
                "$volume": "volume",
            }
        )
        import talib
        fs = talib.get_function_groups()
        tt.fit(X_train, y_train,
            indicators=['all'],
            ranges=[(4, 30)],
            trials=500,
            early_stop=100,
            min_target_correlation=.05,
        )
        
        # Show time duration in seconds per indicator
        tt.fit_times()

        # Show correlation of indicators to target
        tt.report(target_corr=True, features_corr=True)

        # Select features with at most x correlation between each other
        tt.prune(max_inter_correlation=.7)

        # Show correlation of indicators to target and among themselves
        tt.report(target_corr=True, features_corr=True)

        # Add indicators to X_train
        features = tt.transform(X_train)
        X_train = pd.concat([X_train, features], axis=1)

        # Add same indicators to X_test
        features = tt.transform(X_test)
        X_test = pd.concat([X_test, features], axis=1)
    # try:
    #     btcdata = cache("BTCBUSD.parquet")
    #     del btcdata

    # except:
    #     btcdata = df[df["$symbol"] == "BTCBUSD"].compute()
    #     #btcdata = df.compute()
    #     #btcdata = fixvolume(btcdata)
    #     cache("BTCBUSD", btcdata)
    #     del btcdata
    # try:
    #     ethdata = cache("ETHBUSD.parquet")
    #     del ethdata

    # except:
    #     ethdata = df[df["$symbol"] == "ETHBUSD"].compute()
    #     #ethdata = fixvolume(ethdata)
    #     cache("ETHBUSD", ethdata)
    #     del ethdata
    lep = LocalExpressionProvider()
    # df = df.astype(
    #     {
    #         "$close": "float32",
    #         "$high": "float32",
    #         "$low": "float32",
    #         "$open": "float32",
    #         "$volume": "float32",
    #         "$vwap": "float32",
    #         "$adj": "float32",
    #     }
    # )
    scores = {}
    pbar=  tqdm(desc="Processing data",total=(len(dates)*len(symbols)))
    en = 0
    for start_time, end_time in dates:
        substart = (dateparser.parse(start_time) - timedelta(minutes=40330)).strftime("%Y-%m-%d")
        if nproc != 1:
            with Parallel(n_jobs=nproc,timeout=999999,backend="loky") as parallel:
                results = parallel(
                    delayed(ppfc)(
                        df,symbol,lep, scores, substart, end_time,start_time,fieldlist,label
                    )
                    for symbol in symbols
                )
                
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
if __name__ == "__main__":
    compiledata("data",append=True)
