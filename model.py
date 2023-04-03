import os
from datetime import datetime, timedelta

import dateparser
import get_fields

from dask.config import set
from dask.distributed import Client

from subutils import cache, clearcache, loadData
from utils import  procData
from workingdir import WORKING_DIR
import pandas as pd
from preprocessing import nanfill, dist
import numpy as np
import psutil
import math
import os
import pickle as pkl
if __name__ == "__main__":
    app = False
    #import pyarrow as pa
    #print(pa.get_include())
    if not os.path.exists("data.pkl"):
        clearcache()
        if not os.path.exists(WORKING_DIR):
            os.mkdir(WORKING_DIR)
        client = Client()
        nproc = 20
        set(scheduler="distributed", num_workers=nproc)
        labels = [
            "Ref($close, -1)/$close - 1",
            "Ref($close, -5)/$close -1",
            "Ref($close, -10)/$close -1",
            "Ref($close, -30)/$close -1",
            "Ref($close, -60)/$close -1",
            "Ref($close, -360)/$close -1",
            "Ref($close, -1440)/$close -1",
            "Ref($close, -10080)/$close -1",
            "Ref($close, -20160)/$close -1",
            "Ref($close, -40320)/$close -1",
        ]  # label
        label_names = [
            "LABEL1MIN",
            "LABEL5MIN",
            "LABEL10MIN",
            "LABEL30MIN",
            "LABEL60MIN",
            "LABEL360MIN",
            "LABEL1440MIN",
            "LABEL10080MIN",
            "LABEL20160MIN",
            "LABEL40320MIN",
        ]  # label_names'
        
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
        windows = rolling_windows
        fields, names = get_fields.getfields(windows, rolling_windows)

        fields_names = list(map(lambda x, y: (x, y), fields, names))
        labels_names = list(map(lambda x, y: (x, y), labels, label_names))
        
        def date_range(start, end, intv):
            from datetime import datetime
            
            
            diff = (end  - start ) / intv
            for i in range(intv):
                yield (start + diff * i).strftime("%Y-%m-%d %H:%M:%S")
            yield end.strftime("%Y-%m-%d %H:%M:%S")
        start_time = "2022-12-28 00:00:00"
        #start_time = "2022-11-28 00:00:00"
        end_time = "2023-03-28 00:00:00"
        if app:
            
            start_time = pd.read_hdf(os.path.join(WORKING_DIR, "full_data/dset.h5"),"BTCBUSD",start=-1).index[-1].strftime("%Y-%m-%d %H:%M:%S")
            start_time = (dateparser.parse(start_time) - timedelta(minutes=40330)).strftime("%Y-%m-%d %H:%M:%S")
            end_time=pd.read_hdf("C:/Users/Ian/Documents/FT_2/data/dataset.h5","BTCBUSD",start=-1).index[-1].strftime("%Y-%m-%d %H:%M:%S")

        
        start_time_ = (dateparser.parse(start_time) - timedelta(minutes=1))
        end_time_ = (dateparser.parse(end_time) + timedelta(minutes=1))
        if not os.path.exists(os.path.join(WORKING_DIR, "hdf")):
            os.mkdir(os.path.join(WORKING_DIR, "hdf"))
        
        df = loadData(
            start_time=start_time_,
            end_time=end_time_,
            data_dir="C:/Users/Ian/Documents/FT_2/data/dataset.h5",
        )
        dirs = os.listdir(os.path.join(WORKING_DIR,"cache"))
        dirs = [x.split("_")[0] for x in dirs]
        
        symbols = df["$symbol"].compute().unique()
        symbols = [x for x in symbols if x not in dirs]
        
        memusage = df.memory_usage().sum().compute()*len(fields_names)/(len(symbols))
        total_memory = psutil.virtual_memory().total
        memuseperdf = math.ceil(((memusage))/total_memory)
        
        if app:
            start_time_ = (dateparser.parse(start_time) + timedelta(seconds=40330))
        
        dates = date_range(start_time_,end_time_,memuseperdf)
        dates = [x for x in dates]
        dates = [[dates[x],dates[x+1]] for x in range(len(dates)-1)]
        
        df_x = procData(
            df, fields_names, nproc=nproc, dates=dates,symbols=symbols
        )
        # df_y = procData(
        #     df, labels_names, nproc=nproc, dates=dates,label=True,symbols=symbols
        # )
        kel = os.listdir(os.path.join(WORKING_DIR,'cache'))
        df_y_ = {x.split("_")[0]:x for x in kel if 'label' in x}
        df_x = [x for x in kel if 'label' not in x]
        df_x_ = {x.split("_")[0]:x for x in df_x if "cached" in x}
        kel = [x for x in list(df_x_.keys())]
        if not os.path.exists(os.path.join(WORKING_DIR, "full_data")):
            os.mkdir(os.path.join(WORKING_DIR, "full_data"))
        if not app:
            try:
                os.remove(os.path.join(WORKING_DIR, "full_data/dset.h5"))
            except:
                pass
        newv = True
        e = 0
        from sklearn.model_selection import train_test_split
        import pickle as pkl
        dfs = [train_test_split(cache(df_x_[key]).dropna(how="all"),test_size=0.2,shuffle=False) for key in df_x_]
        X_train = [x[0] for x  in dfs]
        X_test = [x[1] for x in dfs]
        y_train = [x["$close"].diff(periods=-60) / x["$close"] * 100 for x in X_train]
        y_test = [x["$close"].diff(periods=-60) / x["$close"] * 100 for x in X_test]
        y_train = pd.concat(y_train)
        y_test = pd.concat(y_test)
        X_train = pd.concat(X_train)
        X_train = X_train.rename(
            columns={
                "$open": "open",
                "$high": "high",
                "$low": "low",
                "$close": "close",
                "$volume": "volume",
            }
        )
        X_test = pd.concat(X_test)
        X_test = X_test.rename(
            columns={
                "$open": "open",
                "$high": "high",
                "$low": "low",
                "$close": "close",
                "$volume": "volume",
            }
        )
        y_train.index = pd.MultiIndex.from_tuples(((pd.to_datetime(x),y) for x,y in zip(X_train.index.values, X_train["$symbol"].values)),names=["date","symbol"])
        y_test.index = pd.MultiIndex.from_tuples(((pd.to_datetime(x),y) for x,y in zip(X_test.index.values, X_test["$symbol"].values)),names=["date","symbol"])
        
        X_train.index = pd.MultiIndex.from_tuples(((pd.to_datetime(x),y) for x,y in zip(X_train.index.values, X_train["$symbol"].values)),names=["date","symbol"])
        X_test.index = pd.MultiIndex.from_tuples(((pd.to_datetime(x),y) for x,y in zip(X_test.index.values, X_test["$symbol"].values)),names=["date","symbol"])
        
        X_train = X_train.drop(["$symbol","$vwap"],axis=1).dropna(how="all")
        X_test = X_test.drop(["$symbol","$vwap"],axis=1).dropna(how="all")
        
        y_train = y_train.dropna(how="all").astype(np.float32)
        y_test = y_test.dropna(how="all").astype(np.float32)
        X_train = X_train.reindex(index=y_train.index).astype(np.float32)
        X_test = X_test.reindex(index=y_test.index).astype(np.float32)
        
        with open("data.pkl","wb") as f:
            pkl.dump((X_train,y_train,X_test,y_test),f)
    else:
        with open("data.pkl","rb") as f:
            X_train,y_train,X_test,y_test = pkl.load(f)
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
    tt = TuneTA(n_jobs=12, verbose=True)
    
    
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
    with open("data_new.pkl","wb") as f:
        pkl.dump((X_train,y_train,X_test,y_test),f)
    print('2')
    # for key in df_x_:
        
    #     df_y = cache(df_y_[key]).dropna(how="all")
    #     loaded_df = cache(df_x_[key]).dropna(how="all")
    #     if app:
    #         prevdata = pd.read_hdf(os.path.join(WORKING_DIR, "full_data/dset.h5"),key,start=-1).index[-1]
    #         df_y = df_y[prevdata + timedelta(minutes=1):]
    #         loaded_df = loaded_df[prevdata + timedelta(minutes=1):]
    #     if not df_y.empty or not loaded_df.empty:
    #         if not app:
    #             df_y = df_y.parallel_apply(nanfill, axis=0)
    #             loaded_df = loaded_df.parallel_apply(nanfill, axis=0)
    #         if not newv:
    #             loaded_df = loaded_df.parallel_apply(dist,args=(app,key,), axis=0).astype(np.float32)
    #             loaded_df = pd.concat({"X": loaded_df, "y": df_y}, axis=1).astype(np.float32)
    #         else:
    #             loaded_df = pd.concat({"X": loaded_df, "y": df_y}, axis=1)
    #         #loaded_df = dd.from_pandas(loaded_df,chunksize=40000)
    #         import time
    #         t = time.time()
    #         if app:
                
                
                
    #             store = pd.HDFStore(os.path.join(WORKING_DIR, "full_data/dset.h5"))
    #             if not loaded_df.empty:
    #                 store.append(key,loaded_df,complevel=1,complib='blosc:zlib')
    #             store.close()
    #             #loaded_df.to_hdf(os.path.join(WORKING_DIR, "full_data/dset.h5"), key,complevel=1,complib='blosc:zlib',mode='a')
    #         else:
    #             loaded_df.to_hdf(os.path.join(WORKING_DIR, "full_data/dset.h5"), key,complevel=1,complib='blosc:zlib',format='table')
    #         t -= time.time()
    #     print(f"{key} took {t} seconds")
           
            #print(t)
        #os.remove(os.path.join(WORKING_DIR, f"cache/{df_x_[key]}"))
        #os.remove(os.path.join(WORKING_DIR, f"cache/{df_y_[key]}"))
        
    #pp.run()
    #print("e")
