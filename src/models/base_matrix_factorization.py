import pickle
from pathlib import Path

import polars as pl

from src.cache import get_polars_hash, get_dict_hash
from src.logger import logger


class BaseMatrixFactorization:
    def __init__(self, cache_dir: Path | None, **kwargs):
        self.cache_dir = cache_dir
        self.cache_params = kwargs

    def _get_hash(
        self, 
        train_df: pl.DataFrame,
        items_meta_df_flatten: pl.DataFrame | None = None,
        users_meta_df_flatten: pl.DataFrame | None = None,
    ): 
        train_df_hash = get_polars_hash(train_df, check_order=False)

        if items_meta_df_flatten is not None:
            items_meta_df_flatten_hash = get_polars_hash(items_meta_df_flatten, check_order=False)
        else:
            items_meta_df_flatten_hash = '0'

        if users_meta_df_flatten is not None:
            users_meta_df_flatten_hash = get_polars_hash(users_meta_df_flatten, check_order=False)
        else:
            users_meta_df_flatten_hash = '0'

        hash_value = get_dict_hash({
            "train_df_hash": train_df_hash,
            "items_meta_df_flatten_hash": items_meta_df_flatten_hash,
            "users_meta_df_flatten_hash": users_meta_df_flatten_hash,
            **self.cache_params,
        })

        logger.debug(f"Hash value: {hash_value}")

        return hash_value

    def _try_load_cache(
        self, 
        train_df: pl.DataFrame,
        items_meta_df_flatten: pl.DataFrame | None = None,
        users_meta_df_flatten: pl.DataFrame | None = None,
    ):
        hash_value = self._get_hash(train_df, items_meta_df_flatten, users_meta_df_flatten)
        logger.debug(f"Hash value: {hash_value}")

        cache_path = self._get_cache_path(hash_value)
        logger.debug(f"Cache path: {cache_path}")

        if cache_path is None:
            return False
        
        if cache_path.exists():
            self.load(cache_path)
            logger.debug(f"Loaded cache")
            return True
        
        return False

    def _save_cache(
        self, 
        train_df: pl.DataFrame,
        items_meta_df_flatten: pl.DataFrame | None = None,
        users_meta_df_flatten: pl.DataFrame | None = None,
    ):
        hash_value = self._get_hash(train_df, items_meta_df_flatten, users_meta_df_flatten)
        cache_path = self._get_cache_path(hash_value)

        if cache_path is not None:
            self.dump(cache_path)

        logger.debug("Cannot save cache, cache_path is None")

    def _get_cache_path(self, hash_value: str) -> Path | None:
        if self.cache_dir is None:
            return None
        
        return self.cache_dir / f"{hash_value}.pickle"

    def dump(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, path: Path):
        with open(path, "rb") as f:
            self_dict = pickle.load(f)

        self.__dict__.update(self_dict) 