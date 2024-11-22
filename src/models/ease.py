from pathlib import Path

import polars as pl
from tqdm import tqdm
import numpy as np
from rectools.models.ease import EASEModel as EASEModel_rectools
from rectools.dataset import Dataset

from src.logger import logger
from src.models.base_matrix_factorization import BaseMatrixFactorization


class EASEModel(BaseMatrixFactorization):
    num_threads = 8

    def __init__(
        self,
        regularization: float = 500.0,
        max_items: int = 30000,
        predict_col_name: str = "predict",
        cold_predict: float = -1.0,
        random_state: int = 42,
        cache_dir: Path | None = None,
    ):
        self.predict_col_name = predict_col_name
        self.cold_predict = cold_predict
        self.regularization = regularization
        self.random_state = random_state
        self.max_items = max_items

        super().__init__(
            cache_dir=cache_dir,
            # params
            predict_col_name=predict_col_name,
            cold_predict=cold_predict,
            regularization=regularization,
            random_state=random_state,
            max_items=max_items,
        )

        self.model = None
        self.item_id_map = None
        self.is_fitted = False

    def _preprocess_train_interactions_df(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        user_id, item_id, weight, datetime

        фильтрация по низкочастотных item
        """
        return (
            df
            .group_by("item_id")
            .len()
            .sort("len")
            .tail(self.max_items)
            .select("item_id")
            .join(df, how="left", on="item_id")
        )

    def fit(
        self, 
        train_df: pl.DataFrame,
        items_meta_df_flatten: pl.DataFrame | None = None,
        users_meta_df_flatten: pl.DataFrame | None = None,
    ):
        train_df = self._preprocess_train_interactions_df(train_df)
        
        load_cache_result = self._try_load_cache(train_df, items_meta_df_flatten, users_meta_df_flatten)
        if load_cache_result:
            return self
        
        dataset = Dataset.construct(interactions_df=train_df.to_pandas())
        self.item_id_map = dataset.item_id_map
        
        self.model = EASEModel_rectools(num_threads=self.num_threads, regularization=self.regularization)
        self.model.fit(dataset)

        self.is_fitted = True

        self._save_cache(train_df, items_meta_df_flatten, users_meta_df_flatten)

        return self
    
    def _get_user2item_sim(self, pairs_df: pl.DataFrame, interactions_df: pl.DataFrame) -> pl.DataFrame:
        assert self.model is not None
        assert self.item_id_map is not None

        ease_predictions = []
        n_batches = 100

        for i in tqdm(range(n_batches)):
            dd = (
                interactions_df.filter(pl.col("user_id") % n_batches == i)
                .join(pairs_df.filter(pl.col("user_id") % n_batches == i), on="user_id", how="inner", suffix="_test")
            )
            
            dd = (
                dd
                .with_columns(
                    ease_weight=self.model.weight[
                        self.item_id_map.convert_to_internal(dd["item_id"].to_numpy()),
                        self.item_id_map.convert_to_internal(dd["item_id_test"].to_numpy()),
                    ]
                )
                .group_by("user_id", pl.col("item_id_test").alias("item_id"))
                .agg(
                    pl.col("ease_weight").sum().alias(self.predict_col_name)
                )
            )
            
            ease_predictions.append(dd)

        ease_predictions = pl.concat(ease_predictions)

        return ease_predictions
    
    def predict_proba(self, pairs_df: pl.DataFrame, interactions_df: pl.DataFrame) -> pl.DataFrame:
        assert self.is_fitted
        assert self.model is not None
        assert self.item_id_map is not None

        assert "user_id" in pairs_df.columns
        assert "item_id" in pairs_df.columns

        filtered_items_df = pl.DataFrame({"item_id": self.item_id_map.external_ids})
        hot_pairs = (
            filtered_items_df
            .join(pairs_df, how="inner", on="item_id")
        )
        filtered_interactions = filtered_items_df.join(interactions_df, how="inner", on="item_id")

        hot_pairs = self._get_user2item_sim(hot_pairs, filtered_interactions)

        cold_pairs = (
            pairs_df
            .join(hot_pairs.select("user_id", "item_id"), how="anti", on=["user_id", "item_id"])
            .with_columns(
                pl.lit(self.cold_predict).cast(pl.Float32).alias(self.predict_col_name)
            )
        )

        logger.info(f"ALS: Percent of cold pairs: {float(cold_pairs.shape[0]) / pairs_df.shape[0]}")

        return pl.concat([hot_pairs, cold_pairs])
