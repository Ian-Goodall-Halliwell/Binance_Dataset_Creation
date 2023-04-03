import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from factor_analyzer import Rotator, calculate_bartlett_sphericity, calculate_kmo
from joblib import Parallel, delayed
from pandarallel import pandarallel
from scipy import stats
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from workingdir import WORKING_DIR
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans, DBSCAN,OPTICS, Birch,BisectingKMeans,SpectralClustering
from sklearn.datasets import load_iris
from ppca import PPCA
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler
from sklearn.mixture import GaussianMixture
np.random.seed(0)


def nanfill(df: pd.DataFrame,nm) -> pd.DataFrame:
    """
    This function fills the NaN values in the dataframe with the mean of the column.
    It also removes the first 40320 rows of the dataframe.
    Args:
        df: The dataframe to be processed.
        name: The name of the dataframe.
    Returns:
        The processed dataframe.
    """
    import numpy as np
    dfout = pd.DataFrame()
    for col in df.columns:
        
        df[col] = df[col].replace([np.inf], df[col].loc[df[col] != np.inf].max())
        df[col] = df[col].replace([-np.inf], df[col].loc[df[col] != -np.inf].min())
        df[col] = df[col].interpolate(method="linear", limit_direction="forward")
        # mean = df.mean(skipna=True)
        # df = df.fillna(mean)
        # df=df.ffill().bfill()
        # df = df[40320:]
        # df = df[:-40320]
        #df[col] = df[col].ffill().bfill()
        #df[col].name = nm + "_" + col
        dfout[nm + "_" + col] = df[col].ffill().bfill()
        nas = df[col].isna().sum() / len(df[col]) * 100
        print(df[col].name, nas)
        if nas > 0:
            print(df[col].name, nas)
    return dfout


def nanfillser(df: pd.Series) -> pd.Series:  # type: ignore
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
    # mean = df.mean(skipna=True)
    # df = df.fillna(mean)
    # df=df.ffill().bfill()
    # df = df[40320:]
    # df = df[:-40320]
    df = df.ffill().bfill()
    nas = df.isna().sum() / len(df) * 100
    # print(df.name, nas)
    if nas > 0:
        print(df.name, nas)
    return df


def clean_data() -> None:
    """
    This function cleans the data.

    Args:
        None

    Returns:
        None
    """
    pandarallel.initialize(progress_bar=True, nb_workers=8)
    store = pd.HDFStore(os.path.join(WORKING_DIR, "full_data/dset.h5"))
    cols = store.keys()
    store.close()
    for col in cols:
        df = pd.read_hdf(
            os.path.join(WORKING_DIR, "full_data/dset.h5"), key=col
        ).astype(np.float32)
        df = df.parallel_apply(nanfill, axis=0)
        counts = df.isna().sum().sort_values()
        print(counts)
        df.to_hdf(os.path.join(WORKING_DIR, "full_data/clean_data.h5"), col)
# N = total # of rows to collate
from itertools import chain
def fast_flatten(input_list):
    a=list(chain.from_iterable(input_list))
    a += [False] * (N - len(a)) # collating logical arrays - missing values are replaced with False
    return list(a)

def combine_lists(frames):
    COLUMN_NAMES = [frames[i].name for i in range(len(frames))]
    COL_NAMES=COLUMN_NAMES
    df_dict = dict.fromkeys(COL_NAMES, [])
    for col in COL_NAMES:
        extracted = (frame[col] for frame in frames)
        df_dict[col] = fast_flatten(extracted)
    Df_new = pd.DataFrame.from_dict(df_dict)[COL_NAMES]
    return Df_new 
def loadpkl(path) -> pd.DataFrame:
    """
    This function concatenates a list of dataframes into one dataframe.

    Args:
        dfs: A list of dataframes.

    Returns:
        A dataframe.
    """
    dirs = os.listdir(path)
    df = pd.DataFrame()
    subls = []
    for dfnm in dirs:
        subpth = os.path.join(path,dfnm)
        subls.append(subpth)
    dfs = [pd.read_pickle(subpthd) for subpthd in subls]
    #del dfs[29]
    #df = combine_lists(dfs)
    dfvals = [dff.values.reshape(-1,1) for dff in dfs]
    odf = pd.DataFrame(
        np.concatenate(dfvals, axis=1),   
        index=dfs[0].index, 
        columns=[df.name for df in dfs]
    )
    #df = pd.concat(dfs, axis=1,join='outer')
    return odf
    

def bypickle(dfs: list) -> pd.DataFrame:
    """
    This function concatenates a list of dataframes into one dataframe.

    Args:
        dfs: A list of dataframes.

    Returns:
        A dataframe.
    """
    df = pd.concat(dfs, axis=1)
    # os.remove(os.path.join(WORKING_DIR,'df_all.pkl'))
    return df


def proc(col: str) -> pd.Series:
    """
    This is a multi-line Google style docstring.

    Args:
        col (str): The column name.

    Returns:
        pd.Series: The series.
    """
    import swifter
    from swifter import set_defaults

    set_defaults(
        progress_bar=False,
    )
    nm = col.split("/")[-1]
    # if c == "BTCBUSD":
    #     return
    # if c == "ETHBUSD":
    #     return
    tempf = pd.read_hdf(os.path.join(WORKING_DIR, "full_data/dset.h5"), key=col)["y"]
    nas = tempf.isna().sum() / len(tempf) * 100
    nas = nas.sum() / len(nas)
    if nas == 100:
        return
    tempf = nanfill(tempf,nm)
    # tempf = tempf.swifter.apply(nanfill, axis=1)
    tempf = tempf.dropna(how="all")
    # tempf = tempf.stack().reset_index(drop=True)
    # df[c] = tempf
    # try:
    #     tempf = tempf.stack()
    # except:
    #     pass
    tempf.name = nm
    if len(tempf) != 0:
        return tempf


if __name__ == "__main__":
    if os.path.exists(os.path.join(WORKING_DIR, "full_data/tempset.pkl")):
        df = pd.read_pickle(os.path.join(WORKING_DIR, "full_data/tempset.pkl"))
    else:
        store = pd.HDFStore(os.path.join(WORKING_DIR, "full_data/dset.h5"))
        cols = store.keys()
        store.close()
        df = []
        c = []
        if os.path.exists(os.path.join(WORKING_DIR, "df_all.h5")):
            os.remove(os.path.join(WORKING_DIR, "df_all.h5"))
        with Parallel(n_jobs=10, backend="loky") as parallel:
            df = parallel(delayed(proc)(col) for col in tqdm(cols))
        df = pd.concat(df, axis=1,join="outer")
        df.to_pickle(os.path.join(WORKING_DIR, "full_data/tempset.pkl"))
    #df = df.T
    # df = loadpkl(os.path.join(WORKING_DIR, "concatcache"))
    # df.to_pickle(os.path.join(WORKING_DIR, "df_clean.pkl"))
    # if os.path.exists(os.path.join(WORKING_DIR, "df_clean.pkl")):
    #     df = pd.read_pickle(os.path.join(WORKING_DIR, "df_clean.pkl"))
    # else:
    dfs = []
    dfdict = pd.DataFrame(index=df.index)
    colvold = df.columns[0].split("_")[0]
    for col in tqdm(df.columns):
        colv = col.split("_")[0]
        if colvold != colv:
            
            dfdict = dfdict.stack(dropna=False)
            dfdict.name = colvold
            dfdict = dfdict.reset_index(level=1,drop=True)
            #dfdict = pd.concat([dfdict, intermframe], axis=1,join='outer')
            dfdict.to_pickle(os.path.join(WORKING_DIR, "concatcache",f"{colvold}.pkl"))
            dfdict = pd.DataFrame(index=df.index)
            #intermframe =pd.DataFrame()
            colvold = colv
        #dfs.append(df[col])
        df_ = pd.Series(RobustScaler().fit_transform(df[col][-110000:].values.reshape(-1, 1)).squeeze(),index=df[col][-110000:].index)
        df_ = df[col][-110000:]
        dfdict = pd.concat([dfdict,df_],axis=1,join='outer')
                
    df = loadpkl(os.path.join(WORKING_DIR, "concatcache"))
    df.to_pickle(os.path.join(WORKING_DIR, "df_clean.pkl"))
    df = df.dropna(how="all")
    df = df.drop(df.columns[df.isna().sum() > 0], axis=1)
    cols = df.columns

    
    df = df.T
    model = PCA(whiten=True)
    model.fit(df)
    output = model.transform(df)
    ev = model.explained_variance_
    kval = len([x for x in ev if x > 1])
    print(kval)
    output = RobustScaler().fit_transform(output)
   # kval = 2
   # output = df
    #kval = 5
    # model = PCA(n_components=kval)
    # # rot = Rotator()
    # model.fit(df)
    scoredict = []
    for xv in tqdm(range(1,kval)):
        tempoutput = output[:,:xv+1]
        limit = int((tempoutput.shape[0] // 2) ** 0.5)
        print(tempoutput.shape)
        
        tempoutput = pd.DataFrame(tempoutput).ffill().bfill().values
        scores = []
        for k in range(1, limit * 8):
            model = AgglomerativeClustering(n_clusters=k)
            model.fit(tempoutput)
            pred = model.fit_predict(tempoutput)#.reshape(-1,1)
            try:
                score = silhouette_score(tempoutput, pred)
            except:
                score = 0
            scores.append(score)
        #print(scores)
        max_value = max(scores)
        
        index = scores.index(max_value) + 1
        scoredict.append([xv+1,max_value,index])
    prevmax = 0
    for x in scoredict:
        if x[1] > prevmax:
            prevmax = x[1]
            bestval = x[0]
            bestidx = x[2]
    model = AgglomerativeClustering(n_clusters=bestidx)
    print(scoredict)
    model.fit(output[:,:bestval])
    pred = model.fit_predict(output[:,:bestval])
    u_labels = np.unique(pred)
    # plotting the results:
    import shutil

    if os.path.exists("clusters"):
        shutil.rmtree("clusters")
        os.mkdir("clusters")
    else:
        os.mkdir("clusters")
    for en in range(output[:,:bestval].shape[1]):
        for i in u_labels:
            out = pred == i
            out = out.flatten()
            dx = output[:, en]
            dx = dx[out]
            dy = output[:, en + 1]
            dy = dy[out]
            plt.scatter(dx, dy, label=i)
        plt.legend()
        plt.savefig(f"clusters/clusters_{en}.png")
        plt.close()
    print("e")
# dendrogram_2017 = sch.dendrogram(sch.linkage(data_2017_extracted_std, method='ward'))
# plt.axhline(y=3.5, color='r', linestyle='--')
