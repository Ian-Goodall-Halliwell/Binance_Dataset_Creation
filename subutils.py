import importlib
import os
import pickle as pkl
import re
import shutil
import sys
from types import ModuleType
import dask.dataframe as dd
import pandas as pd
from workingdir import WORKING_DIR
from typing import Tuple

def split_module_path(module_path: str) -> Tuple[str, str]:
    """
    Parameters
    ----------
    module_path : str
        e.g. "a.b.c.ClassName"
    Returns
    -------
    Tuple[str, str]
        e.g. ("a.b.c", "ClassName")
    """
    *m_path, cls = module_path.split(".")
    m_path = ".".join(m_path)
    return m_path, cls


def loadData(
    start_time: pd.Timestamp, end_time: pd.Timestamp, data_dir: str
) -> dd.DataFrame:
    """
    Load data from hdf5 file

    Args:
        start_time: start time of the data
        end_time: end time of the data
        data_dir: directory of the data

    Returns:
        DataFrame: dataframe of the data
    """
    df = dd.read_hdf(data_dir, "/*")
    df = df.loc[start_time:end_time]
    # df = df.compute()
    return df


def clearcache() -> None:
    """
    Clears the cache directory.
    """
    try:
        cachedir = os.path.join(WORKING_DIR, "cache")
        shutil.rmtree(cachedir)
        os.mkdir(cachedir)
    except:
        pass


def cache(name: str, df=None) -> str:
    if not os.path.exists(os.path.join(WORKING_DIR, "cache")):
        os.mkdir(os.path.join(WORKING_DIR, "cache"))
    cachedir = os.path.join(WORKING_DIR, "cache")
    if df is None:
        if name.split(".")[-1] == "parquet":
            df = pd.read_parquet(os.path.join(cachedir, name))
        else:
            with open(os.path.join(cachedir, name), "rb") as f:
                df = pkl.load(f)
            # df = dd.read_parquet(name)
        return df
    else:
        if isinstance(df, pd.DataFrame):
            name = name.split(".")[0] + ".parquet"
            df.to_parquet(os.path.join(cachedir, name))
        else:
            with open(os.path.join(cachedir, name), "wb") as f:
                pkl.dump(df, f)
        return name


def get_module_by_module_path(module_path: str) -> ModuleType:
    """Load module path
    :param module_path:
    :return:
    :raises: ModuleNotFoundError
    """
    if module_path is None:
        raise ModuleNotFoundError("None is passed in as parameters as module_path")
    if isinstance(module_path, ModuleType):
        module = module_path
    else:
        if module_path.endswith(".py"):
            module_name = re.sub(
                "^[^a-zA-Z_]+",
                "",
                re.sub("[^0-9a-zA-Z_]", "", module_path[:-3].replace("/", "_")),
            )
            module_spec = importlib.util.spec_from_file_location(
                module_name, module_path
            )
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = module
            module_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_path)
    return module


def get_callable_kwargs(config, default_module):
    """
    extract class/func and kwargs from config info
    Parameters
    ----------
    config : [dict, str]
        similar to config
        please refer to the doc of init_instance_by_config
    default_module : Python module or str
        It should be a python module to load the class type
        This function will load class from the config['module_path'] first.
        If config['module_path'] doesn't exists, it will load the class from default_module.
    Returns
    -------
    (type, dict):
        the class/func object and it's arguments.
    Raises
    ------
        ModuleNotFoundError
    """
    if isinstance(config, dict):
        key = "class" if "class" in config else "func"
        if isinstance(config[key], str):
            # 1) get module and class
            # - case 1): "a.b.c.ClassName"
            # - case 2): {"class": "ClassName", "module_path": "a.b.c"}
            m_path, cls = split_module_path(config[key])
            if m_path == "":
                m_path = config.get("module_path", default_module)
            module = get_module_by_module_path(m_path)
            # 2) get callable
            _callable = getattr(module, cls)  # may raise AttributeError
        else:
            _callable = config[key]  # the class type itself is passed in
        kwargs = config.get("kwargs", {})
    elif isinstance(config, str):
        # a.b.c.ClassName
        m_path, cls = split_module_path(config)
        module = get_module_by_module_path(default_module if m_path == "" else m_path)
        _callable = getattr(module, cls)
        kwargs = {}
    else:
        raise NotImplementedError(f"This type of input is not supported")
    return _callable, kwargs
