import dask.dataframe as dd
from workingdir import WORKING_DIR
import pandas as pd
import os
import numpy as np
import lightgbm
import dask.dataframe as dd
from sklearn.model_selection import cross_validate
from flaml import AutoML
from mrmr import mrmr_regression
from pandarallel import pandarallel
import sklearn

def nanfill(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function fills the NaN values in the dataframe with the mean of the column.
    It also removes the first 40320 rows of the dataframe.

    Args:
        df: The dataframe to be processed.

    Returns:
        The processed dataframe.
    """
    import numpy as np

    df = df.replace([np.inf], df.loc[df != np.inf].max())
    df = df.replace([-np.inf], df.loc[df != -np.inf].min())
    df = df.interpolate(method="linear", limit_direction="forward")
    #mean = df.mean(skipna=True)
    #df = df.fillna(mean)
    #df=df.ffill().bfill()
    nas = df.isna().sum() / len(df) * 100
    print(df.name, nas)
    if nas > 0:
        print(df.name, nas)
    df = df[40320:]
    
    df=df.ffill().bfill()
    return df
def loadfunc(col):
    df = pd.read_hdf(os.path.join(WORKING_DIR, "full_data/dset.h5"),key = col).astype(np.float32)
    colsf = [x[0]+x[1] for x in df.columns.values]

    df = df.T.reset_index(drop=True).T
    df.columns = colsf
    df = dd.from_pandas(df,chunksize=100000)
    print(f"Done {col}")
    return df
from joblib import Parallel,delayed
from dask.config import set
from dask.distributed import Client
if __name__ == "__main__":
    client = Client()
    nproc = 32
    set(scheduler="distributed", num_workers=nproc)
    pandarallel.initialize(progress_bar=True, nb_workers=8)
    store = pd.HDFStore(os.path.join(WORKING_DIR, "full_data/dset.h5"))

    cols = store.keys()
    store.close()
    if os.path.exists('experimentlog.txt'):
        os.remove('experimentlog.txt')
    #parts = [client.submit(loadfunc, col) for col in cols]
    with Parallel(n_jobs=nproc,backend='loky') as parallel:
        dfs = parallel(
            delayed(loadfunc)(
                col) for col in cols
            )
        
    # dfs = Parallel()(delayed())
    # for col in cols:
    #     df = pd.read_hdf(os.path.join(WORKING_DIR, "full_data/dset.h5"),key = col).astype(np.float32)
    #     df = dd.from_pandas(df,chunksize=100000)
    #     dfs.append(df)
    print('Starting concat')
    import dask
    #df = dd.from_delayed(parts,meta=loadfunc(cols[0]).head())
    df = dd.concat(dfs, axis=0)
    #df = pd.concat([loadfunc(col) for col in cols], axis=0)
    #df = df.dropna(how="any")
    
    #df = df.compute()
    #df.columns = df.columns.astype(str)
    df = df.compute()
    df.to_parquet(os.path.join(WORKING_DIR, "full_data/long"))
    
    counts = df.isna().sum().sort_values()
    print(counts)

    
    
    
    for i in df['y'].columns:
        
        
        y = df['y'][i]
        df_ = df["X"]
        df_['label'] = y
        feats = mrmr_regression(df_,y,100)
        df_= df_[feats]
        df_['datetime']=pd.to_datetime(df_.index, infer_datetime_format=True)
        cols = df_.columns.tolist()
        cols.remove('datetime')
        cols.insert(0,'datetime')
        df_ = df_[cols]
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(df_, y, test_size=0.2, random_state=42,shuffle=False)
        df_ = X_train
        df_['label'] = y_train
        settings = {
            "time_budget": 100,  # total running time in seconds
            "metric": "rmse",  # primary metrics can be chosen from: ['accuracy','roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'f1','log_loss','mae','mse','r2']
            "task": "ts_forecast",  # task type
            "verbose":10,
            "log_file_name": "experiment2.log",  # flaml log file
            "eval_method": "holdout",
            "log_type": "better",
            "label":'label',
            "dataframe":df,
            "estimator_list":['lgbm', 'rf', 'xgboost', 'extra_tree', 'xgb_limitdepth',"prophet"]
        }
        model = AutoML(**settings)
        counts = df.isna().sum().sort_values()
        print(counts)
        model.fit(dataframe=df_,label='label',period=1000)
        #preds = model.predict(X_test,y_test)
        from flaml.ml import sklearn_metric_loss_score
        #print('train mse', '=', sklearn_metric_loss_score('mse', model.predict(X_train), y_train))
        preds = model.predict(X_test)
        err = sklearn_metric_loss_score('rmse', preds, y_test)
        print(' test mse', '=', err)
        with open('experimentlog.txt', 'a') as f:
            f.write(col+' ' + i + ' ' + str(err) + '\n')
        #scores = cross_validate(model,df['X'][feats],df['y'][i],verbose=2,scoring='neg_mean_squared_error',n_jobs=8)
        print(model.best_result)
        