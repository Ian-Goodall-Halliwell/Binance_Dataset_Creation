import os
import shutil

import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm

from pandarallel import pandarallel

from workingdir import WORKING_DIR

matplotlib.use("Agg")
WORKING_DIR1 = "D://binance"

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
    
    df=df.ffill().bfill()
    nas = df.isna().sum() / len(df) * 100
    #print(df.name, nas)
    if nas > 0:
        print(df.name, nas)
        if nas > 95: 
            df = df.fillna(0)
    df = df[int(40320*1):]
    return df


def dist(X: pd.Series) -> pd.Series:
    """
    This function takes a pandas series and returns a pandas series.

    Args:
        X (pd.Series): A pandas series.

    Returns:
        pd.Series: A pandas series.

    """
    import math
    import numpy as np
    import pandas as pd
    from KDEpy.FFTKDE import FFTKDE
    from matplotlib import pyplot as plt
    from scipy import stats
    from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

    makefigs = True
    nval = -144000
    name = f"figures_hist/{X.name}"
    X = X.diff().fillna(X.mean())
    if makefigs:
        raw = X.hist(bins=100, label="Raw Data")
        plot_open = raw.get_figure()
        plot_open.savefig(f"figures_raw/{X.name}.png")
        plt.close(plot_open)
    outval = X.values.reshape(-1, 1)
    outval = (
        QuantileTransformer(n_quantiles=10000, output_distribution="normal")
        .fit(outval[nval:].reshape(-1, 1))
        .transform(outval.reshape(-1, 1))
        .flatten()
    )
    vals = outval[nval:].flatten()
    kde = True
    if kde:
        kde = FFTKDE(bw="silverman").fit(vals)
        blank = np.linspace(min(outval) - 1e-6, max(outval) + 1e-6, len(outval))
        evaled = kde.evaluate(blank)
        cdf = np.cumsum(evaled)
        cdf *= 1 / cdf.max()
        counts, bins = np.histogram(vals.reshape(-1, 1), bins=100)
        scaledcdf = MinMaxScaler(feature_range=(0, max(counts),)).fit_transform(
            cdf.reshape(-1, 1)
        )
        scaledkde_pdf = MinMaxScaler(feature_range=(0, max(counts),)).fit_transform(
            evaled.reshape(-1, 1)
        )
        best = "kde"
        cdf_b = np.vstack((blank, cdf))

        def find_nearest(value, array=None):
            cdfs = array[1, :]
            array = array[0, :]
            idx = np.searchsorted(array, value, side="left")
            if idx > 0 and (
                idx == len(array)
                or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
            ):
                return cdfs[idx - 1]
            else:
                return cdfs[idx]

        vectorf = np.vectorize(find_nearest, excluded=["array"])(outval, array=cdf_b)
        cdf = pd.Series(vectorf, index=X.index, name=X.name)
        return cdf.astype(np.float32)
    #print(best)
    if makefigs:
        plot = cdf.plot()
        plot_open = plot.get_figure()
        plot_open.savefig(f"figures/{X.name}.png")
        plt.close(plot_open)
        plt.plot(blank, scaledcdf)
        plt.plot(blank, scaledkde_pdf)
        valsmax = max(vals)
        valsmin = min(vals)
        plt.hist(vals.reshape(-1, 1), bins=100)
        plt.xlim(valsmin, valsmax)
        plt.ylim(0, max(counts))
        plt.savefig(f"{name}.png")
        plt.close()
    return vals


def run() -> None:
    """
    This function is used to run the code.
    """
    pandarallel.initialize(progress_bar=False, nb_workers=6)
    # client = Client(timeout=999999)
    nproc = 2
    # set(scheduler="distributed", num_workers=nproc)
    store = pd.HDFStore(os.path.join(WORKING_DIR, "full_data/dset.h5"))
    cols = store.keys()
    store.close()
    try:
        os.remove(os.path.join(WORKING_DIR1, "full_data/dset_processed.h5"))
    except:
        pass
    for col in tqdm(cols):
        for nm in ["figures", "figures_hist", "figures_raw", "acf"]:
            try:
                shutil.rmtree(nm)
                os.mkdir(nm)
            except:
                pass
        # if col != "/LTCBUSD":
        # continue
        df = pd.read_hdf(
            os.path.join(WORKING_DIR, "full_data/dset.h5"), key=col
        ).dropna(how="all")
        #counts = df.isna().sum()
        #print(counts.sort_values())
        df_y = df.xs("y", level=0, axis=1).parallel_apply(nanfill, axis=0)
        df = df.xs("X", level=0, axis=1).parallel_apply(nanfill, axis=0)
        #print(df.isna().sum().sum())
        #pandarallel.initialize(progress_bar=False,nb_workers=6)
        df_y = df_y.parallel_apply(dist, axis=0).astype(np.float32)
        df = df.parallel_apply(dist, axis=0).astype(np.float32)
        df = pd.concat({"X": df, "y": df_y}, axis=1).astype(np.float32)
        # df = dd.from_pandas(df,chunksize=40000)
        
        df.to_hdf(
            os.path.join(WORKING_DIR1, "full_data/dset_processed.h5"), key=col.split("/")[-1],complevel=1,
        )
        # with pd.HDFStore(os.path.join(WORKING_DIR, "full_data/dset.h5"), mode='a') as store:
        #     del store[col]


if __name__ == "__main__":
    run()
