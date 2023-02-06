from subutils import cache
import os
import pandas as pd
import dask.dataframe as dd
from workingdir import WORKING_DIR
import numpy as np
kel = os.listdir('F:/binance_data/cache')
df_y = {x.split("_")[0]:x for x in kel if 'label' in x}
df_x = [x for x in kel if 'label' not in x]
df_x = {x.split("_")[0]:x for x in df_x if "cached" in x}
kel = [x for x in list(df_x.keys())]

# store = pd.HDFStore(os.path.join(WORKING_DIR, "full_data/dset.h5"))

# cols = store.keys()
# store.close()
# for col in cols:
#     df = pd.read_hdf(os.path.join(WORKING_DIR, "full_data/dset.h5"),key = col).astype(np.float32)
#     df.to_hdf(os.path.join(WORKING_DIR, "full_data/dset_tmp.h5"), col.split("/")[-1])
for key in kel:
    
    loaded_df = pd.concat({'X':cache(df_x[key]), 'y':cache(df_y[key])}, axis=1).astype(np.float32)
    #loaded_df = dd.from_pandas(loaded_df,chunksize=40000)
    os.remove(os.path.join(WORKING_DIR + "/cache", df_x[key]))
    os.remove(os.path.join(WORKING_DIR + "/cache", df_y[key]))
    loaded_df.to_hdf(os.path.join(WORKING_DIR, "full_data/dset.h5"), key)