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
from pandarallel import pandarallel
from tqdm import tqdm

if __name__ == "__main__":
   
    # clearcache()
    if not os.path.exists(WORKING_DIR):
       os.mkdir(WORKING_DIR)
    client = Client()
    nproc = 32
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
    start_time = "2021-01-01 00:00:00"
    #start_time = "2022-11-28 00:00:00"
    end_time = "2023-02-06 00:00:00"

    
    start_time_ = (dateparser.parse(start_time) - timedelta(minutes=1))
    end_time_ = (dateparser.parse(end_time) + timedelta(minutes=1))
    if not os.path.exists(os.path.join(WORKING_DIR, "hdf")):
        os.mkdir(os.path.join(WORKING_DIR, "hdf"))
    
    df = loadData(
        start_time=start_time_,
        end_time=end_time_,
        data_dir=os.path.join(WORKING_DIR, "hdf/dataset.h5"),
    )
    dirs = os.listdir(os.path.join(WORKING_DIR,"cache"))
    dirs = [x.split("_")[0] for x in dirs]
    
    symbols = df["$symbol"].compute().unique()
    symbols = [x for x in symbols if x not in dirs]
    # memusage = df.memory_usage().sum().compute()*len(fields_names)/len(symbols)
    # total_memory = psutil.virtual_memory().total
    # memuseperdf = math.ceil(((memusage))/total_memory)
    
    # dates = date_range(start_time_,end_time_,memuseperdf)
    # dates = [x for x in dates]
    # dates = [[dates[x],dates[x+1]] for x in range(len(dates)-1)]
    
    # df_x = procData(
    #     df, fields_names, nproc=nproc, dates=dates,symbols=symbols
    # )
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
    try:
        os.remove(os.path.join(WORKING_DIR, "full_data/dset.h5"))
    except:
        pass
    pandarallel.initialize(nb_workers=64)
     
    for key in tqdm(df_x_):
        
        df_y = cache(df_y_[key]).dropna(how="all")
        loaded_df = cache(df_x_[key]).dropna(how="all")
        if not df_y.empty or not loaded_df.empty:
            try:
                df_y = df_y.parallel_apply(nanfill, axis=0)
                loaded_df = loaded_df.parallel_apply(nanfill, axis=0)
                #df_y = df_y.parallel_apply(dist, axis=0).astype(np.float32)
                loaded_df = loaded_df.parallel_apply(dist, axis=0).astype(np.float32)
                loaded_df = pd.concat({"X": loaded_df, "y": df_y}, axis=1).astype(np.float32)
                #loaded_df = dd.from_pandas(loaded_df,chunksize=40000)
                import time
                t = time.time()
                loaded_df.to_hdf(os.path.join(WORKING_DIR, "full_data/dset.h5"), key,complevel=1,complib='blosc:zlib')
                t -= time.time()
            except:
                pass
            #print(t)
        #os.remove(os.path.join(WORKING_DIR, f"cache/{df_x_[key]}"))
        #os.remove(os.path.join(WORKING_DIR, f"cache/{df_y_[key]}"))
        
    #pp.run()
    #print("e")
