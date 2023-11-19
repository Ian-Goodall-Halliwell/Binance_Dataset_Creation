
def backtest(data,originaldata):
    
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