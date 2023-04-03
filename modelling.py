import dask.dataframe as dd
from workingdir import WORKING_DIR
import pandas as pd
import os
import numpy as np
import lightgbm
import dask.dataframe as dd
from sklearn.model_selection import cross_validate
from flaml import AutoML
# from mrmr import mrmr_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
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
import datetime
import pickle as pkl
from npy_append_array import NpyAppendArray
def loadfuncfull(col):
    df = pd.read_hdf(os.path.join(WORKING_DIR, "full_data/dset.h5"),key = col)#.astype(np.float32)
    df = df.dropna(how="any")
    df_x = df["X"]
    df = df["y"]
    # colsf = [col+"_"+str(x.astype(datetime.datetime)) for x in df_x.index.values]

    # df_x = df_x.reset_index(drop=True)
    # df_x.index = colsf
    # #df.columns = pd.MultiIndex.from_tuples([(x[0],x[1:]) for x in df.columns])
    # #df_x = dd.from_pandas(df_x,chunksize=50000)
    
    
    # colsf = [col+"_"+str(x.astype(datetime.datetime)) for x in df.index.values]
    # df = df.reset_index(drop=True)
    # df.index = colsf
    # #df.columns = pd.MultiIndex.from_tuples([(x[0],x[1:]) for x in df.columns])
    # #df = dd.from_pandas(df,chunksize=50000)
    # cols = df_x.columns
    # with open(os.path.join(WORKING_DIR, "full_data", "cols.pkl"), "wb") as f:
    #     pkl.dump(cols, f)
    
    # filename = os.path.join(WORKING_DIR, "full_data/long_full_x",'out.npy')

    # with NpyAppendArray(filename) as npaa:
    #     npaa.append(np.ascontiguousarray(df_x.values))
        
    # filename = os.path.join(WORKING_DIR, "full_data/long_full_y",'out.npy')

    # with NpyAppendArray(filename) as npaa:
    #     npaa.append(np.ascontiguousarray(df.values))
        
    
    # try:
    #     df_x.to_parquet(os.path.join(WORKING_DIR, "full_data/long_full_x"),append=True,ignore_divisions=True)
    #     df.to_parquet(os.path.join(WORKING_DIR, "full_data/long_full_y"),append=True,ignore_divisions=True)
    # except Exception as e:
    #     print(e)
    #     df_x.to_parquet(os.path.join(WORKING_DIR, "full_data/long_full_x"))
    #     df.to_parquet(os.path.join(WORKING_DIR, "full_data/long_full_y"))
        
    print(f"Done {col}")
def loadfunc(col,cluster):
    df = pd.read_hdf(os.path.join(WORKING_DIR, "full_data/dset.h5"),key = col).astype(np.float32)
    colsf = [x[0]+x[1] for x in df.columns.values]

    df = df.T.reset_index(drop=True).T
    df.columns = colsf
    df = dd.from_pandas(df,chunksize=100000)
    try:
        df.to_parquet(os.path.join(WORKING_DIR, "full_data/long", str(cluster)),append=True)
    except:
        df.to_parquet(os.path.join(WORKING_DIR, "full_data/long", str(cluster)))
        
    print(f"Done {col}")
    #return df
from joblib import Parallel,delayed 
from dask.config import set
from dask.distributed import Client
import pickle as pkl
from lightgbm import LGBMRegressor
if __name__ == "__main__":
    client = Client()
    nproc = 1
    set(scheduler="distributed", num_workers=nproc)
    pandarallel.initialize(progress_bar=True, nb_workers=8)
    store = pd.HDFStore(os.path.join(WORKING_DIR, "full_data/dset.h5"))
    with open("clusterlabels.pkl","rb") as fp:
        labellist = pkl.load(fp)
    cols = store.keys()
    labellist = cols
    store.close()
    if os.path.exists('experimentlog.txt'):
        os.remove('experimentlog.txt')
    #parts = [client.submit(loadfunc, col) for col in cols]
    mkdata =False
    if mkdata:
        for cluster in labellist:
            if os.path.exists(os.path.join(WORKING_DIR, "full_data/long", str(cluster))):
                
                if len(os.listdir(os.path.join(WORKING_DIR, "full_data/long", str(cluster)))) != 0:
                    continue
            
            with Parallel(n_jobs=nproc,backend='loky') as parallel:
                dfs = parallel(
                    delayed(loadfunc)(
                        col,cluster) for col in labellist[cluster]
                    )
            
            print('Starting concat')
            #df = dd.concat(dfs, axis=0)
            #df = df.compute().astype(np.float32)
            df = dd.read_parquet(os.path.join(WORKING_DIR, "full_data/long", str(cluster)))
            #df.to_parquet(os.path.join(WORKING_DIR, "full_data/long", str(cluster)))
    mkdatafull =True
    if mkdatafull:
        
        #loadfuncfull(cols[4])
                    
                
            
        
        with Parallel(n_jobs=nproc,backend='loky') as parallel:
            dfs = parallel(
                delayed(loadfuncfull)(
                    cluster) for cluster in cols
                )
            
        #     print('Starting concat')
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
        #X_train = df.compute().reset_index(drop=True)
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
    exit()
    for i in range(df_y.shape[1]):
        
        if i != 3:
            continue
        y = pd.Series(df_y[:,i])
        y_t = pd.Series(ytest[:,i])
        y *= 100
        y_t *= 100
        stds = df_y[:,i].std(axis=0)
        mean = df_y[:,i].mean(axis=0)
        
        print(stds,mean)
        lgb = True
        if lgb:
            params = {
                "learning_rate": 0.05,
                #"num_leaves": 511,
                #"colsample_bytree": 0.969,
                #"subsample": 0.557,
                "verbosity": -1,
                "random_state": 42,
                'device_type': "gpu",
                "objective": "regression",
                "metric": "l2",
                "num_threads": 22,
                #"min_sum_hessian_in_leaf": 0.1629,
                #"reg_alpha": 5.233545064704743e-07,
                #"reg_lambda": 1.0965560071271672,
                "n_estimators": 1000,
            }
            #model = LGBMRegressor(**params)
            from sklearn.linear_model import LinearRegression, BayesianRidge,ElasticNetCV
            from sklearn.neural_network import MLPRegressor
            from sklearn.svm import SVR
            #from pmlearn.gaussian_process import SparseGaussianProcessRegressor
            model = SVR()
            #model = SparseGaussianProcessRegressor()
            # import GPy
            # kernel = GPy.kern.RBF(1)
            # model = GPy.models.GPRegression(X, Y, kernel=kernel)
            
            model.fit(df_,y)
            preds = model.predict(xtest)
            from sklearn.metrics import r2_score
            # lower = LGBMRegressor(objective = 'quantile', alpha = 1 - 0.75)
            # lower.fit(df_, y)
            # lower_pred = lower.predict(xtest)
            
            # upper = LGBMRegressor(objective = 'quantile', alpha = 0.75)
            # upper.fit(df_, y)
            # upper_pred = upper.predict(xtest)
            
            score = r2_score(y_t, preds)
            print(score)
            
            from flaml.ml import sklearn_metric_loss_score
            err = sklearn_metric_loss_score('mae', preds, y_t)
            print(' test mae', '=', err)
            err = sklearn_metric_loss_score('rmse', preds, y_t)
            print(' test rmse', '=', err)
            with open("savedmodel_gp.pkl","wb") as f:
                pkl.dump((model,model,model),f)
            exit()
            featimps = model.feature_importances_
            featimps = {x:y for x,y in zip(df_.columns.to_list(),featimps) if y >5}
            featimps = dict(sorted(featimps.items(), key=lambda item: item[1],reverse=True))
        #ffs = mrmr_regression(df_, y, K=100)
        tuner = LGBMTuner(metric = 'mae',trials=100,device_type="gpu",verbosity=1,visualization=True) # <- the only required argument
        tuner.fit(df_, y)
        preds = tuner.predict(xtest)
        with open("savedmodel.pkl","wb") as f:
            pkl.dump(tuner.fitted_model,f)
        from flaml.ml import sklearn_metric_loss_score
        #
        err = sklearn_metric_loss_score('rmse', preds, y_t)
        print(' test mse', '=', err)
        err = sklearn_metric_loss_score('mae', preds, y_t)
        print(' test mae', '=', err)
        with open('experimentlog.txt', 'a') as f:
            f.write(str(i)+ ' ' + str(err) + '\n' + str(stds) + '\n' + str(mean) + '\n')
            for key, value in tuner.best_params.items(): 
                f.write('%s:%s\n' % (key, value))