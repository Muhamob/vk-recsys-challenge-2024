from pathlib import Path

import polars as pl
from gensim.models import Word2Vec
from tqdm import tqdm
import numpy as np

from src.logger import logger
from src.models.base_matrix_factorization import BaseMatrixFactorization


def dot_product(x):
    left_embeddings = x.struct[0].to_numpy()
    right_embeddings = x.struct[1].to_numpy()

    dot_product = (left_embeddings * right_embeddings).sum(axis=1)

    return dot_product


class W2VModel(BaseMatrixFactorization):
    num_threads = 8

    def __init__(
        self,
        n_features: int = 128,
        n_epochs: int = 10,
        predict_col_name: str = "predict",
        cold_predict: float = -1.0,
        random_state: int = 42,
        cache_dir: Path | None = None,
    ):
        self.predict_col_name = predict_col_name
        self.cold_predict = cold_predict
        self.n_features = n_features
        self.n_epochs = n_epochs
        self.random_state = random_state

        self.model = None

        super().__init__(
            cache_dir=cache_dir,
            # params
            predict_col_name=predict_col_name,
            cold_predict=cold_predict,
            n_features=n_features,
            n_epochs=n_epochs,
            random_state=random_state,
        )

        self.item_embeddings_df = None
        self.catalog = None

        self.is_fitted = False

    def fit(
        self, 
        train_df: pl.DataFrame,
        items_meta_df_flatten: pl.DataFrame | None = None,
        users_meta_df_flatten: pl.DataFrame | None = None,
    ):
        load_cache_result = self._try_load_cache(train_df, items_meta_df_flatten, users_meta_df_flatten)
        if load_cache_result:
            return self
        
        self.model = Word2Vec(
            (
                train_df
                .group_by("user_id")
                .agg(pl.col("item_id"))
                ["item_id"]
                .to_list()
            ),
            epochs=self.n_epochs,
            vector_size=self.n_features,
            workers=self.num_threads,
            seed=self.random_state,
        )

        self.item_embeddings_df = pl.DataFrame({
            "embeddings": pl.Series(self.model.wv.vectors),
            "item_id": self.model.wv.index_to_key
        }).with_columns(pl.col("item_id").cast(pl.UInt32))

        self.is_fitted = True

        self._save_cache(train_df, items_meta_df_flatten, users_meta_df_flatten)

        return self
    
    @staticmethod
    def calc_mean_embedding(embeddings: pl.Series):
        return np.mean(embeddings[0].to_numpy(), axis=0)
    
    def get_user_embeddings(self, positive_interactions) -> pl.DataFrame:
        n_batches = 1000
        user_embeddings = []
        
        for i in tqdm(range(n_batches), total=n_batches):
            user_embeddings.append((
                positive_interactions
                .filter(pl.col("user_id") % n_batches == i)
                .join(self.item_embeddings_df, how="inner", on="item_id")
                .group_by("user_id")
                .agg(pl.map_groups(exprs=["embeddings"], function=self.calc_mean_embedding))
            ))

        return pl.concat(user_embeddings)  # [user_id, embeddings]
    
    def predict_proba(self, pairs_df: pl.DataFrame, interactions_df: pl.DataFrame) -> pl.DataFrame:
        assert self.is_fitted
        assert self.item_embeddings_df is not None
        assert "user_id" in pairs_df.columns
        assert "item_id" in pairs_df.columns

        user_embeddings = self.get_user_embeddings(interactions_df)
        item_embeddings = self.item_embeddings_df

        hot_pairs = (
            pairs_df
            .join(user_embeddings.select("user_id"), how="inner", on="user_id")
            .join(item_embeddings.select("item_id"), how="inner", on="item_id")
        )

        cold_pairs = (
            pairs_df
            .join(hot_pairs.select("user_id", "item_id"), how="anti", on=["user_id", "item_id"])
            .with_columns(
                pl.lit(self.cold_predict).cast(pl.Float32).alias(self.predict_col_name)
            )
        )

        n_batches = 1000
        hot_pairs_predict = []
        for i in tqdm(range(n_batches), total=n_batches):
            sample_df = hot_pairs.filter(pl.col("user_id") % n_batches == i)
            hot_pairs_predict.append((
                sample_df
                .join(user_embeddings.rename({"embeddings": "user_embeddings"}), how="left", on="user_id")
                .join(item_embeddings.rename({"embeddings": "item_embeddings"}), how="left", on="item_id")
                .with_columns(pl.struct("user_embeddings", "item_embeddings").map_batches(dot_product).alias(self.predict_col_name))
                .drop("user_embeddings", "item_embeddings")
            ))

        hot_pairs_predict = pl.concat(hot_pairs_predict)

        logger.info(f"ALS: Percent of cold pairs: {float(cold_pairs.shape[0]) / pairs_df.shape[0]}")

        return pl.concat([hot_pairs_predict, cold_pairs])


class W2VSourceModel(W2VModel):
    def add_item_df(self, items_meta: pl.DataFrame):
        self.items_meta = items_meta
        return self
    
    def fit(
        self, 
        train_df: pl.DataFrame,
        items_meta_df_flatten: pl.DataFrame | None = None,
        users_meta_df_flatten: pl.DataFrame | None = None,
    ):
        assert self.items_meta is not None

        df = (
            train_df
            .join(self.items_meta.select("item_id", "source_id"), how="inner", on="item_id")
            .select("user_id", pl.col("source_id").alias("item_id"))
        )

        return super().fit(df, items_meta_df_flatten, users_meta_df_flatten)
    
    def predict_proba(self, pairs_df: pl.DataFrame, interactions_df: pl.DataFrame) -> pl.DataFrame:
        assert self.items_meta is not None

        pairs_df = (
            pairs_df
            .join(self.items_meta.select("item_id", "source_id"), on="item_id", how="left")
            .select("user_id", pl.col("source_id").alias("item_id"), pl.col("item_id").alias("raw_item_id"))
        )

        interactions_df = interactions_df.join(self.items_meta.select("item_id", "source_id"), on="item_id", how="left").drop("item_id").rename({"source_id": "item_id"})

        return super().predict_proba(pairs_df, interactions_df).select("user_id", pl.col("raw_item_id").alias("item_id"), self.predict_col_name)
