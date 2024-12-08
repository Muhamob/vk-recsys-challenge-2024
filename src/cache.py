import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
from src.logger import logger


def get_pandas_hash(df: pd.DataFrame, check_order: bool = False) -> str:
    rows_hash = pd.util.hash_pandas_object(df, index=True).values

    if check_order:
        return str(int(hashlib.sha256(rows_hash).hexdigest(), 16))
    else:
        return str(rows_hash.sum())
    

def get_polars_hash(df: pl.DataFrame, check_order: bool = False) -> str:
    rows_hash = df.hash_rows().to_numpy()
    
    if check_order:
        return str(int(hashlib.sha256(rows_hash).hexdigest(), 16))
    else:
        return str(rows_hash.sum())


def get_dict_hash(data: dict[str, Any]) -> str:
    dict_as_str = json.dumps(data, sort_keys=True)
    return str(int(hashlib.sha256(dict_as_str.encode('utf-8')).hexdigest(), 16))


def polars_output_cache(save_dir):
    def wrap(func):
        def wrapped_func(*args):
            hashes = {}
            for i, arg in enumerate(args):
                if isinstance(arg, pl.DataFrame):
                    hashes[f"arg_{i}"] = get_polars_hash(arg, check_order=False)
                elif isinstance(arg, pd.DataFrame):
                    hashes[f"arg_{i}"] = get_pandas_hash(arg, check_order=False)
                else:
                    hashes[f"arg_{i}"] = arg

            hash_value = get_dict_hash(hashes)
            cache_path = Path(save_dir) / f"{hash_value}.parquet"

            if cache_path.exists():
                logger.info(f"Cache found in {cache_path}. Loading data")
                df = pl.read_parquet(cache_path)
            else:
                logger.info(f"No cache found in {cache_path}. Calculating data")
                df = func(*args)
                df.write_parquet(cache_path)

            return df
        return wrapped_func
    return wrap
