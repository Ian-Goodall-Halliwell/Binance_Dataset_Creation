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
from models import pytorch_lstm
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

import pickle as pkl

from joblib import Parallel,delayed 
from dask.config import set
from dask.distributed import Client
import pickle as pkl
from lightgbm import LGBMRegressor
from verstack import LGBMTuner
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
    
    
    with open("data_new.pkl","rb") as f:
        X_train,y_train,X_test,y_test = pkl.load(f)
    # for x in (X_train,X_test,y_train,y_test):
    #     x = x.xs(x.axes[0].levels[1][-1],level=1)
    from sklearn.preprocessing import QuantileTransformer, StandardScaler
    # X_train = X_train.xs(X_train.axes[0].levels[1][3],level=1)
    # X_test = X_test.xs(X_test.axes[0].levels[1][3],level=1)
    # y_train = y_train.xs(y_train.axes[0].levels[1][3],level=1)
    # y_test = y_test.xs(y_test.axes[0].levels[1][3],level=1)
    #X_train = X_train.drop("fta_QSTICK_period_16",axis=1)
    #X_test = X_test.drop("fta_QSTICK_period_16",axis=1)
    print(X_train.isna().sum().sort_values())
    print(X_test.isna().sum().sort_values())
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train = X_train.dropna(how="all")
    X_test = X_test.dropna(how="all")
    
    X_train = X_train.fillna(method='ffill')
    X_test = X_test.fillna(method='ffill')
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    X_train = X_train.dropna(how="any")
    X_test = X_test.dropna(how="any")
    
    #X_train = X_train.fillna(method='bfill')
    #X_test = X_test.fillna(method='bfill')
    y_train = y_train.reindex(X_train.index)
    y_test = y_test.reindex(X_test.index)
    cols = X_train.columns
    
    stds = y_test.std(axis=0)
    mean = y_train.mean(axis=0)
    print(mean)
    stds = y_test.std(axis=0)
    mean = y_test.mean(axis=0)
    y_train = y_train.sort_index()
    y_test = y_test.sort_index()
    X_train = X_train.sort_index()
    
    import imblearn
    ccent=imblearn.under_sampling.RandomUnderSampler()
    #X_train = X_train.drop(["symbol"],axis=1)
    
    binary = False
    if binary == True:
        # y_test = y_test.indices
        # y_train = y_train.indices
        y_train
        yidx = y_train.index.to_numpy().reshape(-1,1)
        yidx,y_train = ccent.fit_resample(yidx,y_train)
        yidx = list(x[0] for x in yidx)
        yidx = pd.MultiIndex.from_tuples(yidx)
        X_train = X_train.reindex(yidx)
        mn = y_train.mean()
        y_train.index=yidx
    else:
        import torch
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import KBinsDiscretizer
        nbin=3
        kb = KBinsDiscretizer(strategy="quantile",n_bins=nbin)
        kb.fit(y_train,)
        y_train = kb.transform(y_train).toarray()
        y_test = kb.transform(y_test).toarray()
        X_train,y_train = ccent.fit_resample(X_train,y_train)
        X_pure, y_pure = X_test,y_test
        X_train, X_test, y_train, y_test = train_test_split(X_train.values,y_train,test_size=0.3)
        X_pure = torch.tensor(X_pure.values, dtype=torch.float32 ).cuda()
        y_pure = torch.tensor(y_pure, dtype=torch.float32 ).cuda()
        #X_test,y_test = ccent.fit_resample(X_test,y_test.toarray())
        
    print(y_train.sum(axis=0))
    print(stds,mean)
    lgb = True
    if lgb:
        params = {
            #"learning_rate": 0.01,
            #"num_leaves": 256,
            #"colsample_bytree": 0.969,
            #"subsample": 0.557,
            "verbosity": -1,
            "random_state": 42,
            'device_type': "cpu",
            "objective": "multiclass",
            "num_class":5,
            "num_threads": 6,
            #"min_sum_hessian_in_leaf": 0.1629,
            #"max_depth":4,
            #"reg_alpha": 0,
            #"reg_lambda": 0.5,
            #"n_estimators": 1000,
        }
        #model = LGBMRegressor(**params)
        from sklearn.linear_model import LogisticRegressionCV,RidgeClassifierCV,SGDClassifier
        from sklearn.naive_bayes import CategoricalNB,GaussianNB,MultinomialNB
        from sklearn.neural_network import MLPRegressor,MLPClassifier
        from sklearn.gaussian_process import GaussianProcessRegressor,GaussianProcessClassifier
        from sklearn.svm import SVR
        from sklearn.metrics import r2_score
        import lightgbm
        #
        from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,f1_score,auc,roc_auc_score
        model = LGBMRegressor(**params)
        tnn = True
        if tnn:
            
            import torch
            import torch.nn as nn
            insize = X_train.shape[1]
            outsize = nbin
            hiddensize=64
            class Multiclass(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.GRU(input_size=insize, hidden_size=hiddensize, num_layers=1, batch_first=True)
                    self.act = nn.ReLU()
                    self.linear = nn.Linear(hiddensize, outsize)
                def forward(self, x):
                    x, _ = self.lstm(x)
                    x = self.act(x)
                    x = self.linear(x)
                    return x
            # class Multiclass(nn.Module):
            #     def __init__(self):
            #         super().__init__()
            #         self.hidden = nn.Linear(insize, hiddensize)
            #         self.act = nn.ReLU()
            #         self.output = nn.Linear(hiddensize, outsize)
                    
            #     def forward(self, x):
            #         x = self.act(self.hidden(x))
            #         x = self.output(x)
            #         return x
                
            model = Multiclass().cuda()
            
            import torch.optim as optim
            import tqdm
            import copy
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            # convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors
            X_train = torch.tensor(X_train, dtype=torch.float32).cuda()
            X_test = torch.tensor(X_test, dtype=torch.float32).cuda()
            y_train = y_train#.toarray()
            y_test = y_test#.toarray()
            y_test = torch.tensor(y_test, dtype=torch.float32 ).cuda()
            y_train = torch.tensor(y_train, dtype=torch.float32 ).cuda()
            
            # training parameters
            n_epochs = 150
            batch_size = 1024*4
            batches_per_epoch = len(X_train) // batch_size
            best_acc = - np.inf   # init to negative infinity
            best_weights = None
            train_loss_hist = []
            train_acc_hist = []
            test_loss_hist = []
            test_acc_hist = []
            
            for epoch in range(n_epochs):
                epoch_loss = []
                epoch_acc = []
                # set model in training mode and run through each batch
                model.train()
                with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
                    bar.set_description(f"Epoch {epoch}")
                    for i in bar:
                        # take a batch
                        start = i * batch_size
                        X_batch = X_train[start:start+batch_size]
                        y_batch = y_train[start:start+batch_size]
                        # forward pass
                        y_pred = model(X_batch)
                        loss = loss_fn(y_pred, y_batch)
                        # backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        # update weights
                        optimizer.step()
                        # compute and store metrics
                        ym = torch.argmax(y_pred, 1)
                        yb = torch.argmax(y_batch, 1)
                        acc = ( ym== yb).float().mean()
                        epoch_loss.append(float(loss))
                        epoch_acc.append(float(acc))
                        bar.set_postfix(
                            loss=float(loss),
                            acc=float(acc)
                        )
                # set model in evaluation mode and run through the test set
                model.eval()
                y_pred = model(X_test)
                ce = loss_fn(y_pred, y_test)
                acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
                ce = float(ce)
                acc = float(acc)
                train_loss_hist.append(np.mean(epoch_loss))
                train_acc_hist.append(np.mean(epoch_acc))
                test_loss_hist.append(ce)
                test_acc_hist.append(acc)
                if acc > best_acc:
                    best_acc = acc
                    best_weights = copy.deepcopy(model.state_dict())
                print(f"Epoch {epoch} validation: Cross-entropy={ce}, Accuracy={acc}")
                
            model.load_state_dict(best_weights)
            
            import matplotlib.pyplot as plt
            
            plt.plot(train_loss_hist, label="train")
            plt.plot(test_loss_hist, label="test")
            plt.xlabel("epochs")
            plt.ylabel("cross entropy")
            plt.legend()
            plt.show()
            
            plt.plot(train_acc_hist, label="train")
            plt.plot(test_acc_hist, label="test")
            plt.xlabel("epochs")
            plt.ylabel("accuracy")
            plt.legend()
            plt.show()
            preds = model(X_pure)
            preds = preds.cpu().detach().numpy()
            y_pure = y_pure.cpu().detach().numpy()
            
        else:
            y_train = y_train.indices
            y_test = y_test.indices
            model = MLPClassifier(hidden_layer_sizes=500,max_iter=100)
       
            model.fit(X_train,y_train)
            preds = model.predict(X_test)
        
        score = roc_auc_score(y_pure, preds)
        preds = [np.argmax(x) for x in preds]
        y_pure = [np.argmax(x) for x in y_pure]
        import matplotlib.pyplot as plt
        #preds = [1 if x > 0.5 else 0 for x in preds ]
        matrix = confusion_matrix(y_pure,preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
        disp.plot()
        plt.show()
        f1 = f1_score(y_pure,preds,average="micro")
        print(' test f1', '=', f1)
        #plot = lightgbm.plot_tree(model)
        #plot.plot()
        #plot.figure.savefig("plot.png",dpi=1000)
        print(str(score) + " " + "full")
        for v in range(X_train.shape[1]):
            #model = LGBMRegressor(**params)
            model = GaussianProcessClassifier()
            model.fit(X_train.values[:,v].reshape(-1, 1),y_train.values)
            preds = model.predict(X_test.values[:,v].reshape(-1, 1))
            score = r2_score(y_test.values, preds)
            print(str(score) + " " + str(v))
        
        from flaml.ml import sklearn_metric_loss_score
        err = sklearn_metric_loss_score('mae', preds, y_test.values)
        print(' test mae', '=', err)
        err = sklearn_metric_loss_score('rmse', preds, y_test.values)
        print(' test rmse', '=', err)
        with open("savedmodel_gp.pkl","wb") as f:
            pkl.dump(model,f)
        exit()
        featimps = model.feature_importances_
        featimps = {x:y for x,y in zip(df_.columns.to_list(),featimps) if y >5}
        featimps = dict(sorted(featimps.items(), key=lambda item: item[1],reverse=True))
        
    from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,f1_score,auc    
    tuner = LGBMTuner(metric = 'f1',trials=50,device_type="cpu",verbosity=1,visualization=False) # <- the only required argument
    tuner.fit(X_train, y_train.squeeze())
    preds = tuner.predict(X_test)
    with open("savedmodel.pkl","wb") as f:
        pkl.dump(tuner.fitted_model,f)
    from flaml.ml import sklearn_metric_loss_score
    from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,f1_score
    import matplotlib.pyplot as plt
    #
    matrix = confusion_matrix(y_test.squeeze(),preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot()
    plt.show()
    score = auc(y_test, preds)
    f1 = f1_score(y_test.squeeze(),preds)
    print(' test f1', '=', f1)
    err = sklearn_metric_loss_score('rmse', preds, y_test.squeeze())
    print(' test mse', '=', err)
    err = sklearn_metric_loss_score('mae', preds, y_test.squeeze())
    print(' test mae', '=', err)
    with open('experimentlog.txt', 'a') as f:
        f.write(str(i)+ ' ' + str(err) + '\n' + str(stds) + '\n' + str(mean) + '\n')
        for key, value in tuner.best_params.items(): 
            f.write('%s:%s\n' % (key, value))