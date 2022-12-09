import os
from datetime import datetime, timedelta

import dateparser
import get_fields
import numpy as np
from dask.config import set
from dask.distributed import Client
from statsmodels.tsa.stattools import adfuller
from subutils import cache, clearcache, loadData
from utils import frac_diff, procData
from workingdir import WORKING_DIR
import pandas as pd
if __name__ == "__main__":
    clearcache()
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)
    client = Client()
    nproc = 7
    set(scheduler="distributed", num_workers=nproc//2)
    labels = [
        "Ref($close, -1)/$close - 1",
        "Ref($close, -5)/$close -1",
        "Ref($close, -10)/$close -1",
        "Ref($close, -30)/$close -1",
        "Ref($close, -360)/$close -1",
        "Ref($close, -1440)/$close -1",
    ]  # label
    label_names = [
        "LABEL1MIN",
        "LABEL5MIN",
        "LABEL10MIN",
        "LABEL30MIN",
        "LABEL360MIN",
        "LABEL1440MIN",
    ]  # label_names'
    windows = range(14)
    rolling_windows = [
        3,
        6,
        9,
        12,
        24,
        36,
        52,
        72,
        144,
        360,
        720,
        1440,
        2160,
        2880,
        3600,
        4320,
        4760,
        6200,
        7640,
        9080,
        12400,
        19600,
        24800,
    ]
    fields, names = get_fields.getfields(windows, rolling_windows)

    fields_names = list(map(lambda x, y: (x, y), fields, names))
    labels_names = list(map(lambda x, y: (x, y), labels, label_names))
    
    def date_range(start, end, intv):
        from datetime import datetime
        
        
        diff = (end  - start ) / intv
        for i in range(intv):
            yield (start + diff * i).strftime("%Y-%m-%d")
        yield end.strftime("%Y-%m-%d")
    start_time = "2019-01-02 00:00:00"
    #start_time = "2022-11-28 00:00:00"
    end_time = "2022-12-01 00:00:00"

    
    start_time_ = (dateparser.parse(start_time) - timedelta(minutes=1))
    end_time_ = (dateparser.parse(end_time) + timedelta(minutes=1))
    if not os.path.exists(os.path.join(WORKING_DIR, "hdf")):
        os.mkdir(os.path.join(WORKING_DIR, "hdf"))
    dates = date_range(start_time_,end_time_,8)
    dates = [x for x in dates]
    dates = [[dates[x],dates[x+1]] for x in range(len(dates)-1)]
    df = loadData(
        start_time=start_time_,
        end_time=end_time_,
        data_dir=os.path.join(WORKING_DIR, "hdf/dataset.h5"),
    )
    df_x = procData(
        df, fields_names, nproc=nproc, dates=dates
    )
    df_y = procData(
        df, labels_names, nproc=nproc, dates=dates,label=True
    )
    teststationarity = False
    if teststationarity:
        adadict = df_x["ADABUSD"]
        ada = cache(adadict)

        for col in ada.columns:
            mn = ada[col].mean()
            if mn == np.nan:
                mn = 0

            c = ada[col].fillna(mn)
            try:
                ad = frac_diff(c)[0]
                adfresults = adfuller(c)
                if adfresults[1] > 0.05:
                    print(col, adfresults[1])
            except Exception as e:
                print(e)

    if not os.path.exists(os.path.join(WORKING_DIR, "full_data")):
        os.mkdir(os.path.join(WORKING_DIR, "full_data"))
    try:
        os.remove(os.path.join(WORKING_DIR, "full_data/dset.h5"))
    except:
        pass
    for key in df_x:
        loaded_df = pd.concat({'X':cache(df_x[key]), 'y':cache(df_y[key])}, axis=1)

        loaded_df.to_hdf(os.path.join(WORKING_DIR, "full_data/dset.h5"), key)

    print("e")
