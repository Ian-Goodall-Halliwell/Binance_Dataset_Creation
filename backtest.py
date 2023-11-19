from backtesting.test import GOOG

from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd
import pickle as pkl
from datetime import datetime
from workingdir import WORKING_DIR
import os
with open("C:/Users/Ian/Documents/FT_2/savedmodel_gp.pkl",'rb') as f:
    model = pkl.load(f)
# with open(os.path.join(WORKING_DIR, "full_data/dataset_.pkl"),"rb") as f:
#             df_,df_y,xtest,ytest = pkl.load(f)
def SMA(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).rolling(n).mean()
def nullf(values):
    return values
def findlims(values,stdmulti):
    
    
    # values = values[0]
    # with open("C:/Users/Ian/Documents/FT_2/data_new.pkl","rb") as f:
    #     values = pkl.load(f)[1].xs(values[1:],level=1,axis=0)
    
    std = values.std() * stdmulti
    mean = values.mean()
    return (mean,std)
class PredStrat(Strategy):
    stdmulti = 1
    def init(self):
        self.stdmulti = self.stdmulti
        self.preds = self.I(nullf,self.data.Target)
        #self.predslow = self.I(nullf,self.data.Target_lower)
        #self.predshigh = self.I(nullf,self.data.Target_upper)
        name = self.data.name
        values = olddat_
        self.lims = findlims(values,self.stdmulti)
    def next(self):
        if self.preds[-1] > self.lims[0] + self.lims[1]:
            self.position.close()
            self.buy()
        elif self.preds[-1] < self.lims[0] - self.lims[1]:
            self.position.close()
            self.sell()
        else:
            self.position.close()


from backtesting import Backtest
import os
from workingdir import WORKING_DIR
import numpy as np
store = pd.HDFStore(os.path.join(WORKING_DIR, "hdf/dataset.h5"))
cols = store.keys()
store.close()
start_time = pd.Timestamp("2023-03-18 00:00:00")
with open("C:/Users/Ian/Documents/FT_2/data_new.pkl","rb") as f:
    dat = pkl.load(f)
    olddat = dat[0]
    dat = dat[2]
dat.replace([np.inf, -np.inf], np.nan, inplace=True)
dat = dat.dropna(how="all")
dat = dat.fillna(method='ffill')
dat = dat.fillna(0)


olddat.replace([np.inf, -np.inf], np.nan, inplace=True)
olddat = olddat.dropna(how="all")
olddat = olddat.fillna(method='ffill')
olddat = olddat.fillna(0)
#dat = dat.dropna(how="any")
global olddat_

for key in cols:
    #dset_ = pd.read_hdf("C:/Users/Ian/Documents/FT_2/data.pkl",key)['X']
    try:
        dset_ = dat.xs(key[1:],level=1,axis=0)
    except:
        continue
    olddat_ = olddat.xs(key[1:],level=1,axis=0)
    data = pd.read_hdf("C:/Users/Ian/Documents/FT_2/data/dataset.h5",key)[["$open","$high","$low","$close","$volume"]].loc[dset_.index]
    data = data.rename(columns={"$open":"Open","$high":"High","$low":"Low","$close":"Close","$volume":"Volume"})
    if key == "/BTCBUSD":
        data[["Open","High","Low","Close"]] = data[["Open","High","Low","Close"]] / 100000
        data["Volume"] = data["Volume"] * 100000
    data[["High"]] = data[["High"]].resample("60T",label='right').max()
    data[["Low"]] = data[["Low"]].resample("60T",label='right').min()
    data[["Close"]] = data[["Close"]].resample("60T",label='right').last()
    data[["Open"]] = data[["Open"]].resample("60T",label='right').first()
    data["Volume"] = data["Volume"]#
    data["Volume"] = data["Volume"].resample("60T",label='right').sum()
    data = data.loc[start_time:,:]
    data = data.dropna(axis=0,how='any')
    dset_ = dset_.loc[start_time:,:]
    try:
        dset = pd.Series(model[0].predict(dset_.loc[data.index]), index=data.index, name="Target")
        olddat_ = pd.Series(model[0].predict(olddat_), index=olddat_.index, name="Prev")
    except:
        continue
    data = pd.concat([data,dset],axis=1)
    # dset = pd.Series(model[1].predict(dset_.loc[data.index]), index=data.index, name="Target_lower")
    # data = pd.concat([data,dset],axis=1)
    # dset = pd.Series(model[2].predict(dset_.loc[data.index]), index=data.index, name="Target_upper")
    # data = pd.concat([data,dset],axis=1)
    data["name"] = key
    bt = Backtest(data, PredStrat, cash=5000, commission=.0012)
    rng = np.arange(1, 100, 1).tolist()
    rng = [x/10 for x in rng]
    stats, heatmap = bt.optimize(stdmulti=rng,return_heatmap=True,maximize="Return [%]",max_tries=200,constraint=lambda p: p.stdmulti > 0,)
    #stats = bt.run()
    # r1 = 29.5
    print(stats)
    # bt.plot()