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
        with Parallel(n_jobs=4, backend="loky") as parallel:
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
            try:
                dfdict = dfdict.stack(dropna=False)
                dfdict.name = colvold
                dfdict = dfdict.reset_index(level=1,drop=True)
            except Exception as e:
                print(e)
                pass
           # dfdict = dfdict.replace([np.inf], dfdict.loc[dfdict != np.inf].max())
            #dfdict = dfdict.replace([-np.inf], dfdict.loc[dfdict != -np.inf].min())
            #dfdict = pd.concat([dfdict, intermframe], axis=1,join='outer')
            dfdict.to_pickle(os.path.join(WORKING_DIR, "concatcache",f"{colvold}.pkl"))
            dfdict = pd.DataFrame(index=df.index)
            #intermframe =pd.DataFrame()
            colvold = colv
        #dfs.append(df[col])
        #df_ = pd.Series(MinMaxScaler().fit_transform(df[col][-110000:].values.reshape(-1, 1)).squeeze(),index=df[col][-110000:].index)
        df_ = df[col][int(-525600//2):]
        if not df_.isna().sum() > 0:
            #df_ = pd.Series(StandardScaler(with_mean=False).fit_transform(df_.values.reshape(-1, 1)).squeeze(),index=df_.index)
            df_  /= df_.std(axis=0)
            df_ = df_.replace([np.inf], df_.loc[df_ != np.inf].max())
            df_ = df_.replace([-np.inf], df_.loc[df_ != -np.inf].min())
            
        #dfdict = df_
        dfdict = pd.concat([dfdict,df_],axis=1,join='outer')
        #dfdict = df_
        
                
    df = loadpkl(os.path.join(WORKING_DIR, "concatcache"))
    df.to_pickle(os.path.join(WORKING_DIR, "df_clean.pkl"))
    print(df.shape)
    df = df.dropna(how="all")
    df = df.drop(df.columns[df.isna().sum() > 0], axis=1)
    print(df.shape)
    names = df.columns
    variation = df.values.T
    from sklearn import covariance

    #alphas = np.logspace(-100, 1, num=1000)
    edge_model = covariance.GraphicalLassoCV()

    # standardize the time series: using correlations rather than covariance
    # former is more efficient for structure recovery
    X = variation.copy().T
    #X /= X.std(axis=0)
    edge_model.fit(X)
    
    
    from sklearn import cluster

    _, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=0)
    n_labels = labels.max()
    labellist = {}
    for i in range(n_labels + 1):
        print(f"Cluster {i + 1}: {', '.join(names[labels == i])}")
        labellist.update({i:names[labels == i]})
    import pickle as pkl
    with open("clusterlabels.pkl","wb") as fp:
        pkl.dump(labellist,fp)
        
    from sklearn import manifold

    node_position_model = manifold.LocallyLinearEmbedding(
        n_components=2, eigen_solver="dense", n_neighbors=6
    )

    embedding = node_position_model.fit_transform(X.T).T
    
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    plt.figure(1, facecolor="w", figsize=(10, 8))
    plt.clf()
    ax = plt.axes([0.0, 0.0, 1.0, 1.0])
    plt.axis("off")

    # Plot the graph of partial correlations
    partial_correlations = edge_model.precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero = np.abs(np.triu(partial_correlations, k=1)) > 0.02

    # Plot the nodes using the coordinates of our embedding
    plt.scatter(
        embedding[0], embedding[1], s=100 * d**2, c=labels, cmap=plt.cm.nipy_spectral
    )

    # Plot the edges
    start_idx, end_idx = np.where(non_zero)
    # a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [
        [embedding[:, start], embedding[:, stop]] for start, stop in zip(start_idx, end_idx)
    ]
    values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(
        segments, zorder=0, cmap=plt.cm.hot_r, norm=plt.Normalize(0, 0.7 * values.max())
    )
    lc.set_array(values)
    lc.set_linewidths(15 * values)
    ax.add_collection(lc)

    # Add a label to each node. The challenge here is that we want to
    # position the labels to avoid overlap with other labels
    texts = []
    for index, (name, label, (x, y)) in enumerate(zip(names, labels, embedding.T)):

        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = "left"
            x = x + 0.002
        else:
            horizontalalignment = "right"
            x = x - 0.002
        if this_dy > 0:
            verticalalignment = "bottom"
            y = y + 0.002
        else:
            verticalalignment = "top"
            y = y - 0.002
        from adjustText import adjust_text
        texts.append(plt.text(
            x,
            y,
            name,
            size=10,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            bbox=dict(
                facecolor="w",
                edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
                alpha=0.6,
            ),
        ))
    adjust_text(texts)    
    plt.xlim(
        embedding[0].min() - 0.15 * embedding[0].ptp(),
        embedding[0].max() + 0.10 * embedding[0].ptp(),
    )
    plt.ylim(
        embedding[1].min() - 0.03 * embedding[1].ptp(),
        embedding[1].max() + 0.03 * embedding[1].ptp(),
    )

    plt.show()
    plt.savefig("clustmap.png")
    labellist = names[labels == i]
