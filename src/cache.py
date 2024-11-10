import hashlib
import json
from typing import Any

import pandas as pd
import polars as pl


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
