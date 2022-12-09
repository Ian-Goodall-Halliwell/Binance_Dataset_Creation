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
    nproc = -1
    set(scheduler="distributed", num_workers=nproc)
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
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        14,
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

    start_time = "2019-01-01 00:00:00"
    #start_time = "2022-11-28 00:00:00"
    end_time = "2022-12-01 00:00:00"

    start_time_ = (dateparser.parse(start_time) - timedelta(minutes=1)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    end_time_ = (dateparser.parse(end_time) + timedelta(minutes=1)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    if not os.path.exists(os.path.join(WORKING_DIR, "hdf")):
        os.mkdir(os.path.join(WORKING_DIR, "hdf"))
    df = loadData(
        start_time=start_time_,
        end_time=end_time_,
        data_dir=os.path.join(WORKING_DIR, "hdf/dataset.h5"),
    )
    df_x = procData(
        df, fields_names, nproc=nproc, start_time=start_time, end_time=end_time
    )
    df_y = procData(
        df, labels_names, nproc=nproc, start_time=start_time, end_time=end_time,label=True
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
