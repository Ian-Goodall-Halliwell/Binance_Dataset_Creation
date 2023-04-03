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
    df = pd.read_hdf(os.path.join(WORKING_DIR, "full_data/dset.h5"),key = col).astype(np.float32)
    df = df.dropna(how="any")
    df_x = df["X"]
    df = df["y"]
    colsf = [col+"_"+str(x.astype(datetime.datetime)) for x in df_x.index.values]

    df_x = df_x.reset_index(drop=True)
    df_x.index = colsf
    #df.columns = pd.MultiIndex.from_tuples([(x[0],x[1:]) for x in df.columns])
    #df_x = dd.from_pandas(df_x,chunksize=50000)
    
    
    colsf = [col+"_"+str(x.astype(datetime.datetime)) for x in df.index.values]
    df = df.reset_index(drop=True)
    df.index = colsf
    #df.columns = pd.MultiIndex.from_tuples([(x[0],x[1:]) for x in df.columns])
    #df = dd.from_pandas(df,chunksize=50000)
    cols = df_x.columns
    with open(os.path.join(WORKING_DIR, "full_data", "cols.pkl"), "wb") as f:
        pkl.dump(cols, f)
    
    filename = os.path.join(WORKING_DIR, "full_data/long_full_x",'out.npy')

    with NpyAppendArray(filename) as npaa:
        npaa.append(np.ascontiguousarray(df_x.values))
        
    filename = os.path.join(WORKING_DIR, "full_data/long_full_y",'out.npy')

    with NpyAppendArray(filename) as npaa:
        npaa.append(np.ascontiguousarray(df.values))
        
    
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
    mkdatafull =False
    if mkdatafull:
        
            # if os.path.exists(os.path.join(WORKING_DIR, "full_data/long", str(cluster))):
                
            #     if len(os.listdir(os.path.join(WORKING_DIR, "full_data/long", str(cluster)))) != 0:
            #         continue
            
        with Parallel(n_jobs=nproc,backend='loky') as parallel:
            dfs = parallel(
                delayed(loadfuncfull)(
                    cluster) for cluster in cols
                )
            
            print('Starting concat')
            #df = dd.concat(dfs, axis=0)
            #df = df.compute().astype(np.float32)
        
        #df = pd.read_parquet(os.path.join(WORKING_DIR, "full_data/long_full"))
            #df.to_parquet(os.path.join(WORKING_DIR, "full_data/long", str(cluster)))
        

    
    from sklearn.multioutput import MultiOutputRegressor
    from optuna.integration.lightgbm import LightGBMTunerCV
    from verstack import LGBMTuner
    from lightgbm import Dataset, LGBMRegressor
    
    clust = False
    if clust:
        for cluster in labellist:
            df = dd.read_parquet(os.path.join(WORKING_DIR, "full_data/long", str(cluster))).compute()
            df.columns = pd.MultiIndex.from_tuples([(x[0],x[1:]) for x in df.columns])
            print(len(df))
            #continue
            df = df.dropna(how="any")
            counts = df.isna().sum().sort_values()
            print(len(df))
            # print(counts)
        
            for i in df["y"].columns:
                
                y = df['y'][i]
                
                df_ = df["X"]
                #df_['label'] = y
                # mrmr_regression()
                
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(df_, y, test_size=0.2, random_state=42,shuffle=False)
                #df_ = X_train
                #train_data = Dataset(X_train,y_train,params={"max_bin":15})
                #test_data = Dataset(X_test,y_test,reference=train_data,params={"max_bin":15})
                #df_['label'] = y_train
                # settings = {
                #     "time_budget": 1000,  # total running time in seconds
                #     "metric": "rmse",  # primary metrics can be chosen from: ['accuracy','roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'f1','log_loss','mae','mse','r2']
                #     "task": "regression",  # task type
                #     "verbose":0,
                #     "log_file_name": "experiment2.log",  # flaml log file
                #     "eval_method": "cv",
                #     "log_type": "better",
                #     "n_jobs": -1,
                #     "device":"gpu",
                #     "dataframe":df,
                #     "estimator_list":['lgbm']
                # }
                # model = MultiOutputRegressor(AutoML(**settings))
                #model = MultiOutputRegressor(LGBMRegressor())
                tuner = LGBMTuner(metric = 'rmse',trials=10,device_type="gpu",verbosity=0,visualization=False) # <- the only required argument
                tuner.fit(X_train, y_train)
                preds = tuner.predict(X_test)
                # model_ = LightGBMTunerCV(train_set=train_data,return_cvbooster=True)
                # model_.run()
                # model = model_.get_best_booster()
                #
                from flaml.ml import sklearn_metric_loss_score
                #print('train mse', '=', sklearn_metric_loss_score('mse', model.predict(X_train), y_train))
                #preds = model.predict(test_data)
                err = sklearn_metric_loss_score('rmse', preds, y_test)
                print(' test mse', '=', err)
                # with open('experimentlog.txt', 'a') as f:
                #     f.write(str(cluster)+' ' + ' ' + str(err) + '\n')
                #scores = cross_validate(model,df['X'][feats],df['y'][i],verbose=2,scoring='neg_mean_squared_error',n_jobs=8)
                with open('experimentlog.txt', 'a') as f:
                    f.write(str(cluster)+' ' +i+ ' ' + str(err) + '\n' )
                    for key, value in tuner.best_params.items(): 
                        f.write('%s:%s\n' % (key, value))
                #print(model_.best_params)
                #print(model_.best_score)
    else:
        
        if not os.path.exists(os.path.join(WORKING_DIR, "full_data/dataset_.pkl")):
            df_ = np.load(os.path.join(WORKING_DIR, "full_data/long_full_x",'out.npy'), mmap_mode="r")
            df_y = np.load(os.path.join(WORKING_DIR, "full_data/long_full_y",'out.npy'), mmap_mode="r")
            idx = np.random.choice(np.arange(df_y.shape[0]), int(df_y.shape[0]//16), replace=False)
            mask = np.zeros(len(df_), dtype=bool)
            mask[idx] = True
            df_ = df_[mask]
            df_y = df_y[mask]
            means = df_y.mean(axis=0)
            stds = df_y.std(axis=0)
            print(means, stds)
            with open(os.path.join(WORKING_DIR, "full_data", "cols.pkl"), "rb") as f:
                cols = pkl.load(f)
            #df_ = pd.DataFrame(df_,columns=cols)
            indices = np.arange(df_y.shape[0])
            from sklearn.model_selection import train_test_split
            train, test = train_test_split(indices, test_size=0.2, random_state=42,shuffle=True)
            mask_train = np.zeros(len(df_), dtype=bool)
            mask_train[train] = True
            mask_test = np.zeros(len(df_), dtype=bool)
            mask_test[test] = True
            xtest = pd.DataFrame(df_[mask_test],columns=cols,copy=False)
            print("xtest done")
            ytest = df_y[mask_test]
            print("ytest done")
            df_ = pd.DataFrame(df_[mask_train],columns=cols,copy=False)
            print("xtrain done")
            df_y = df_y[mask_train]
            print("ytrain done")
            with open(os.path.join(WORKING_DIR, "full_data/dataset_.pkl"),"wb") as f:
                pkl.dump((df_,df_y,xtest,ytest),f)
        else:
            with open(os.path.join(WORKING_DIR, "full_data/dataset_.pkl"),"rb") as f:
                df_,df_y,xtest,ytest = pkl.load(f)
            means = ytest.mean(axis=0)
            stds = ytest.std(axis=0)
            print(means, stds)
        for i in range(df_y.shape[1]):
            
            if i != 3:
                continue
            y = pd.Series(df_y[:,i])
            y_t = pd.Series(ytest[:,i])
            stds = df_y[:,i].std(axis=0)
            mean = df_y[:,i].mean(axis=0)
            print(stds,mean)
            lgb = True
            if lgb:
                # params = {
                #     "learning_rate": 0.05,
                #     "num_leaves": 473,
                #     "colsample_bytree": 0.7989499894055425,
                #     "subsample": 0.5442462510259598,
                #     "verbosity": -1,
                #     "random_state": 42,
                #     'device_type': "gpu",
                #     "objective": "regression",
                #     "metric": "l1",
                #     "num_threads": 22,
                #     "min_sum_hessian_in_leaf": 0.006080390190296602,
                #     "reg_alpha": 2.5529693461039728e-08,
                #     "reg_lambda": 8.471746987003668e-06,
                #     "n_estimators": 100000,
                # }
                # model = LGBMRegressor(**params)
                # model.fit(df_, y, feature_name=df_.columns.to_list())
                from gpr import ExactGPModel
                import gpytorch
                import torch
                from torch.utils.data import TensorDataset, DataLoader
                batch_size = 1024
                
                
                df_ = torch.tensor(df_.values)
                y = torch.tensor(y.values)
                train_dataset = TensorDataset(df_, y)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)
                
                xtest = torch.tensor(xtest.values)
                y_t = torch.tensor(y_t.values)
                test_dataset = TensorDataset(xtest, y_t)
                test_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)
                
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                model = ExactGPModel(df_,y,likelihood)
                model.fit(train_loader,likelihood)
                preds,var = model.predict(test_loader,likelihood)
                
                
                from flaml.ml import sklearn_metric_loss_score
                err = sklearn_metric_loss_score('mae', preds, y_t)
                print(' test mae', '=', err)
                with open("savedmodel_gp.pkl","wb") as f:
                    pkl.dump(model,f)
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