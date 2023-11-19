from workingdir import WORKING_DIR
import pandas as pd
import os
import numpy as np
import pickle as pkl
import imblearn
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,f1_score,auc,roc_auc_score,r2_score,mean_squared_error
import torch.optim as optim
import tqdm
import copy
from sklearn.linear_model import ElasticNetCV,LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from models import pytorch_lstm,pytorch_nn,double_ensemble, pytorch_tra,pytorch_transformer
import torch
from sklearn.preprocessing import StandardScaler
from modelcode import RNNmodel, TimeSeriesDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


def remove_outliers(data_array, threshold=3):
    # Calculate the mean and standard deviation of the array
    labels = data_array['label'].values
    data_array = data_array['feature']
    #data_array = data_array[-5000:]
    #labels = labels[-5000:]

    mean = np.mean(labels)
    std = np.std(labels)
    
    # Calculate the Z-scores for each data point
    z_scores = np.abs((labels - mean) / std)
    
    # Find the indices of outliers using the threshold
    outlier_indices = np.where(z_scores > threshold)
    
    # Remove the outliers from the original array
    #cleaned_data = np.delete(data_array, outlier_indices,axis=0)
    labels[outlier_indices] = threshold
    return {"feature":data_array,"label":labels}

if __name__ == "__main__":
    
    
    
    with open("data_new.pkl","rb") as f:
        X_train,y_train,X_test,y_test,orders = pkl.load(f)


    #print(X_train.isna().sum().sort_values())
    #print(X_test.isna().sum().sort_values())
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
    

    y_train = y_train.reindex(X_train.index)
    y_test = y_test.reindex(X_test.index)
    cols = X_train.columns
    
    stds = y_test.std(axis=0)
    mean = y_train.mean(axis=0)
    #print(mean)
    stds = y_test.std(axis=0)
    mean = y_test.mean(axis=0)
    y_train = y_train.sort_index()
    y_test = y_test.sort_index()
    X_train = X_train.sort_index()
    

    ccent=imblearn.under_sampling.RandomUnderSampler()
    classs = False
    if classs==True:
        
        binary = False
        if binary == True:
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
    undersample=False
    import random
    if undersample:
        # dates = random.sample(X_train.index.levels[0].to_list(),5000)
        # X_train = X_train[X_train.index.get_level_values('date').isin(dates)]
        # y_train = y_train[y_train.index.get_level_values('date').isin(dates)]
        # dates = random.sample(X_test.index.levels[0].to_list(),1000)
        # X_test = X_test[X_test.index.get_level_values('date').isin(dates)]
        # y_test = y_test[y_test.index.get_level_values('date').isin(dates)]
        
        # X_train = X_train[-50000:]
        # y_train = y_train[-50000:]
        
        # X_test = X_test[:10000]
        # y_test = y_test[:10000]
        pass
    
    #print(y_train.sum(axis=0))
    #print(stds,mean)
    from sklearn.model_selection import KFold
    
    cv = KFold(shuffle=False)
    scores = []
    X_train,y_train,X_test,y_test = X_train.astype(np.float32),y_train.astype(np.float32),X_test.astype(np.float32),y_test.astype(np.float32)
    #for train,test in cv.split(X_train,y_train):
        
    df_train = {"feature":X_train,"label":y_train}
    df_valid = {"feature":X_test,"label":y_test}  
    
    
    def studyfunc(trial):
        scores=[]
        
            
        datacols = df_train["feature"].index.levels[1].to_list()
        # alsodatacols = orders.index.to_list()
        # alsodata = []
        # for ds in alsodatacols:
        #     incl = trial.suggest_categorical(f"{ds}", [True,False])
        #     if incl:
        #         alsodata.append(ds)
        # data = []
        # for ds in datacols:
        #     incl = trial.suggest_categorical(f"{ds}", [True,False])
        #     if incl:
        #         data.append(ds)
        xdf_t = {'feature':df_train["feature"],'label':df_train["label"]}
        xdf_true = {'feature':X_test,'label':y_test}
        
        # xdf_t = {'feature':df_train["feature"][df_train["feature"].index.get_level_values('symbol').isin(data)],'label':df_train["label"][df_train["label"].index.get_level_values('symbol').isin(data)]}
        outlierthresh = 5
        
        # xdf_true = {'feature':X_test[X_test.index.get_level_values('symbol').isin(data)],'label':y_test[y_test.index.get_level_values('symbol').isin(data)]}
        #xdf_t = remove_outliers(xdf_t,threshold=outlierthresh)
        
        
        #input_sequence = torch.randn(batch_size, sequence_length, input_size)
        # Create an instance of the TimeSeriesDataset
        sequence_length = trial.suggest_categorical('seq',[24,48,64,96])
        

        # Define hyperparameters
        input_size = xdf_t['feature'].shape[1] # Replace ... with the input size of your RNN
        hidden_size = trial.suggest_categorical('hidden_size',[32,64,128,256,512])  # Replace ... with the hidden size of your RNN
        output_size = 1  # Replace ... with the output size of your RNN
        learning_rate = trial.suggest_categorical('learning_rate',[0.01,0.001,0.0001,0.00001])
        weight_decay = trial.suggest_categorical('weight_decay',[0,0.00001,0.0001,0.001,0.01])
        dropout = trial.suggest_categorical('dropout',[0,0.1,0.2])
        num_epochs = 100
        batch_size = 1000
        
        layers = trial.suggest_categorical('layers',[1,2,3,4])
        model = RNNmodel(input_size,hidden_size,layers,output_size,4,learning_rate,num_epochs,dropout,weight_decay)
        
        
        
        
        names = y_train.index.levels[1].unique().to_list()
        scores = []
        for stock in names:
            #labelscaler = StandardScaler()
            scaler = StandardScaler()
            y = y_train.xs(stock,axis=0,level=1)
            
            X_t = X_train.xs(stock,axis=0,level=1).values
            xdf_t = remove_outliers({'feature':X_t,'label':y},threshold=outlierthresh)
            # startidx = y_.index.get_level_values('date').unique()[1]
            X_ = scaler.fit_transform(xdf_t["feature"])
            
            y_ = xdf_t['label'][1:].reshape(-1,1)
            X_ = X_[1:]
            #y_ = y_train.values
            #y_ = y_[1:]
            # y_neg  = -np.log(abs(y_train.values[y_train.values < 0]))
            # y_pos  = np.log(y_train.values[y_train.values > 0])
            #y_ = labelscaler.fit_transform(y_)

            
            #X_ = np.where(X_ > outlierthresh, outlierthresh, X_)
            #X_ = np.where(X_ < -outlierthresh, -outlierthresh, X_)

            
            #y_ = np.where(y_ > outlierthresh, outlierthresh, y_)
            #y_ = np.where(y_ < -outlierthresh, -outlierthresh, y_)
            
            if len(y_) <= batch_size*2:
                continue
            score = model.cross_validate(X_,y_,sequence_length,batch_size,outlierthresh)
            print(score)
            scores.append(score)
        # dataset = TimeSeriesDataset(scaler.fit_transform(xdf_t['feature'].values),xdf_t['label'].values, sequence_length)
        # dataset_true = TimeSeriesDataset(scaler.transform(xdf_true['feature'].values),xdf_true['label'].values, sequence_length)
        # dataloader = DataLoader(dataset, batch_size=batch_size,pin_memory=True,pin_memory_device='cuda', shuffle=False)
        # dataloader_true = DataLoader(dataset_true, batch_size=batch_size,pin_memory=True,pin_memory_device='cuda', shuffle=False)
        # model.train(dataloader)
        # preds,all_reals = model.predict(dataloader_true)
        # score = r2_score(all_reals,preds)
        # #scores.append(score)
        #score = model.cross_validate(xdf_t["feature"].values,xdf_t["label"].values,sequence_length,batch_size)
        #score = np.mean(scores)
        top_2_idx = np.argsort(scores)[-5:]
        score = np.mean([scores[i] for i in top_2_idx])
        return score

    import optuna
    study_name = "fintest2"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    initoptim = True
    study = optuna.create_study(storage=storage_name,direction="maximize",study_name="fintest2",load_if_exists=True)
    if initoptim:
        study.optimize(studyfunc,n_jobs=1)
    else:
        bp = study.best_params
        print(f"Best trial value = {study.best_value}")
        datacols = df_train["feature"].index.levels[1].to_list()
        def optimdata():
            model = MLPRegressor(
                hidden_layer_sizes=(100,),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                batch_size="auto",
                learning_rate="constant",
                learning_rate_init=0.001,
                power_t=0.5,
                max_iter=200,
                shuffle=True,
                random_state=None,
                tol=1e-4,
                verbose=False,
                warm_start=False,
                momentum=0.9,
                nesterovs_momentum=True,
                early_stopping=False,
                validation_fraction=0.1,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
                n_iter_no_change=10,
                max_fun=15000,
            )
            # pytorch_transformer.TransformerModel(
            #     d_feat = df_train["feature"].shape[1],
            #     d_model = bp.pop("d_model"),
            #     batch_size = 10000, #trial.suggest_categorical("batch_size",[1000,2000,3000,4000,6000,8000]),
            #     nhead = bp.pop("nhead"),
            #     num_layers = bp.pop("num_layers"),
            #     dropout = bp.pop("dropout"),
            #     n_epochs = 1000,
            #     lr = bp.pop("lr"),
            #     metric = "",
            #     early_stop = 20,
            #     loss = "mse",
            #     optimizer = "adam",
            #     reg = bp.pop("reg"),
            #     n_jobs = 0,
            #     GPU = 0,
            #     seed = 2,
            #     )
            data = []
            for ds in datacols:
                incl = bp.pop(f"{ds}")
                if incl:
                    data.append(ds)
            
            xdf_t = {'feature':df_train["feature"][df_train["feature"].index.get_level_values('symbol').isin(data)],'label':df_train["label"][df_train["label"].index.get_level_values('symbol').isin(data)]}
           
            
            xdf_true = {'feature':X_test[X_test.index.get_level_values('symbol').isin(data)],'label':y_test[y_test.index.get_level_values('symbol').isin(data)]}
            
            
            model.fit(xdf_t,xdf_true)
            with open("savedmodel_gp.pkl","wb") as f:
                pkl.dump([model],f)
            
            preds = model.predict(xdf_true["feature"])
            score = r2_score(xdf_true["label"],preds)
            return score
            #df_t = df_train["feature"].xs(data,level="symbol",drop_level=False)
        optimdata()  
        #study_name = "datatest2"  # Unique identifier of the study.
        #storage_name = "sqlite:///{}.db".format(study_name)
        #study = optuna.create_study(storage=storage_name,direction="maximize",study_name="datatest2",load_if_exists=True)
        #study.optimize(optimdata)
            
    print(study.best_params)
    print(study.best_trial)
    try:
        model.fit(df_train,df_valid)
    except:
        model.fit(df_train["feature"],df_train["label"])
    preds = model.predict(X_test)
    score = r2_score(y_test,preds)
    #scores.append(score)
    #means = np.mean(scores)
    print(score)
    print(score)
    
    
    # score = roc_auc_score(y_test, preds)
    # preds = [np.argmax(x) for x in preds]
    # y_pure = [np.argmax(x) for x in y_test]
    # import matplotlib.pyplot as plt
    # matrix = confusion_matrix(y_pure,preds)
    # disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    # disp.plot()
    # plt.show()
    # f1 = f1_score(y_pure,preds,average="micro")
    # print(' test f1', '=', f1)

    # print(str(score) + " " + "full")
    