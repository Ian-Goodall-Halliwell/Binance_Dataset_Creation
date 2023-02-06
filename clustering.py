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
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.datasets import load_iris
from ppca import PPCA
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.preprocessing import StandardScaler
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
  
    df = pd.concat([pd.read_pickle(subpthd) for subpthd in subls], axis=0,join='outer')
    return df
    

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
        with Parallel(n_jobs=4, backend="loky") as parallel:
            df = parallel(delayed(proc)(col) for col in tqdm(cols))
        df = pd.concat(df, axis=1,join="outer")
        df.to_pickle(os.path.join(WORKING_DIR, "full_data/tempset.pkl"))
    #df = df.T
    df = loadpkl(os.path.join(WORKING_DIR, "concatcache"))
    df.to_pickle(os.path.join(WORKING_DIR, "df_clean.pkl"))
    if os.path.exists(os.path.join(WORKING_DIR, "df_clean.pkl")):
        df = loadpkl(os.path.join(WORKING_DIR, "concatcache"))
    else:
        dfdict = pd.DataFrame(index=df.index)
        colvold = df.columns[0].split("_")[0]
        for col in tqdm(df.columns):
            colv = col.split("_")[0]
            if colvold != colv:
                
                dfdict = dfdict.stack()
                dfdict.name = colvold
                #dfdict = pd.concat([dfdict, intermframe], axis=1,join='outer')
                dfdict.to_pickle(os.path.join(WORKING_DIR, "concatcache",f"{colv}.pkl"))
                dfdict = pd.DataFrame(index=df.index)
                #intermframe =pd.DataFrame()
                colvold = colv
            
            dfdict = pd.concat([dfdict,df[col]],axis=1,join='outer')
                  
        df = loadpkl(os.path.join(WORKING_DIR, "concatcache"))
    df = df.dropna(how="all")
    df = df.drop(df.columns[df.isna().sum() > 0], axis=1)
    cols = df.columns

    df = StandardScaler().fit_transform(df).T
    output = df.T
    model = PCA()
    model.fit(df)
    ev = model.explained_variance_
    kval = len([x for x in ev if x > 1])
    print(kval)
    #kval = 10
    model = PCA(n_components=kval)
    # rot = Rotator()
    model.fit(df)
    output = model.transform(df)

    limit = int((output.shape[0] // 2) ** 0.5)
    # determining number of clusters
    # using silhouette score method
    output = pd.DataFrame(output).ffill().bfill().values
    scores = []
    for k in range(2, limit * 4):
        model = MiniBatchKMeans(n_clusters=k)
        model.fit(output)
        pred = model.predict(output)
        score = silhouette_score(output, pred)
        scores.append(score)
    print(scores)
    max_value = max(scores)
    index = scores.index(max_value) + 1
    model = MiniBatchKMeans(n_clusters=index)
    model.fit(output)
    pred = model.predict(output)
    u_labels = np.unique(pred)
    # plotting the results:
    import shutil

    if os.path.exists("clusters"):
        shutil.rmtree("clusters")
        os.mkdir("clusters")
    else:
        os.mkdir("clusters")
    for en in range(output.shape[0] - 1):
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
