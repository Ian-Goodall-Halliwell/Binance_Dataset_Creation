import os
import pandas as pd
from modelcode import RNNmodel
import numpy as np
#from btestfunc import backtest
def load(path):
    for file in os.listdir(os.path.join(path, "1m-raw")):
        ds = pd.read_csv(os.path.join(os.path.join(path, "1m-raw"), file))
        ds = ds.set_index(pd.to_datetime(ds["date"]))
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
        })
        
        yield ds
        # ds.to_hdf(
        #     os.path.join(path, "dataset.h5"),
        #     key=file.split(".")[0],
        #     mode="a",
        #     format="table",
        # )

def timebin(X,binsize,num_subsets):
    num_subsets -= 1
    step = binsize//num_subsets
    for id in range(num_subsets):
        step_temp = step*id
        if step_temp != 0:
            yield X[:-step_temp].resample(f"{binsize}T",origin='end').last()
        else:
            yield X.resample(f"{binsize}T",origin='end').last()
    #print('')
data_old = {}
for d in load('data'):
    d_name = d['$symbol'].values[0]
    del d['$symbol'], d['$vwap'], d['date']
    #print(d)
    #d = d.diff()
    #print(d)
    # labelidx = d['$close'].index.shift(-1,freq='min')
    # label = d["$close"]
    # label.index = labelidx
    # label = label
    # d = d[2:-1]#.values.astype(np.float32)
    #d = d.values.astype(np.float32)
    # label = label[3:]#.values.astype(np.float32) * 100
    #label = label.values.astype(np.float32) * 100
    # print(label.head(31))
    #print(d)
    # labels = []
    # for l in timebin(label,binsize=30,num_subsets=4):
    #     l = l.diff()
    #     labels.append(l)
        
    # data = []
    # for l in timebin(d,binsize=30,num_subsets=4):
    #     da = l.diff()
    #     data.append(da)
    data_old[d_name] = d



sequence_length = 100
    
import pickle as pkl   

with open("data_new.pkl","rb") as f:
    X_train,y_train,X_test,y_test,t_corr= pkl.load(f)
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
names = y_train.index.levels[1].unique().to_list()
scores = []
for stock in names:
    defaultdata = data_old[stock]
    y = y_train.xs(stock,axis=0,level=1)
    X_t = X_train.xs(stock,axis=0,level=1).values
    
    xdf_t = {'feature':X_t,'label':y}
    # Define hyperparameters
    input_size = xdf_t["feature"].shape[1] # Replace ... with the input size of your RNN
    hidden_size = 200  # Replace ... with the hidden size of your RNN
    output_size = 1  # Replace ... with the output size of your RNN
    learning_rate = 0.0001
    weight_decay = 0
    dropout = 0
    num_epochs = 100
    batch_size = 500
    
    layers = 1
    model = RNNmodel(input_size,hidden_size,layers,output_size,4,learning_rate,num_epochs,dropout,weight_decay)
    score = model.cross_validate(xdf_t["feature"].astype(np.float32),xdf_t["label"].values.astype(np.float32),sequence_length,batch_size,defaultdata)
    
    print(score)
    scores.append(score)
#print(len(X_train),len(y_train))
#score = model.cross_validate(X_train,y_train,sequence_length,batch_size)
print(np.mean(scores))
