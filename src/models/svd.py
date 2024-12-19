from pathlib import Path

import polars as pl
from rectools.models.pure_svd import PureSVDModel
from rectools.dataset import Dataset

from src.logger import logger
from src.models.base_matrix_factorization import BaseMatrixFactorization


class SVDModel(BaseMatrixFactorization):
    num_threads = 8

    def __init__(
        self,
        n_features: int = 10,
        predict_col_name: str = "predict",
        cold_predict: float = -1.0,
        verbose: int = 0,
        random_state: int = 42,
        cache_dir: Path | None = None,
    ):
        self.predict_col_name = predict_col_name
        self.cold_predict = cold_predict
        self.n_features = n_features
        self.random_state = random_state

        self.model = PureSVDModel(self.n_features, verbose=verbose, random_state=random_state)

        super().__init__(
            cache_dir=cache_dir,
            # params
            n_features=n_features,
            random_state=random_state,
        )

        self.user2idx = None
        self.nmid2idx = None
        self.user_embeddings = None
        self.item_embeddings = None

        self.is_fitted = self.model.is_fitted

    def fit(
        self, 
        train_df: pl.DataFrame,
        items_meta_df_flatten: pl.DataFrame | None = None,
        users_meta_df_flatten: pl.DataFrame | None = None,
    ):
        load_cache_result = self._try_load_cache(train_df, items_meta_df_flatten, users_meta_df_flatten)
        if load_cache_result:
            return self
        
        assert self.model is not None
        
        dataset = Dataset.construct(interactions_df=train_df.to_pandas())

        self.user2idx = dataset.user_id_map
        self.item2idx = dataset.item_id_map

        self.model.fit(dataset)
        
        self.user_embeddings, self.item_embeddings = self.model.get_vectors()
        self.is_fitted = self.model.is_fitted

        self._save_cache(train_df, items_meta_df_flatten, users_meta_df_flatten)

        return self
    
    def predict_proba(self, pairs_df: pl.DataFrame) -> pl.DataFrame:
        assert self.is_fitted
        assert self.user2idx is not None
        assert self.item2idx is not None
        assert self.user_embeddings is not None
        assert self.item_embeddings is not None
        assert "user_id" in pairs_df.columns
        assert "item_id" in pairs_df.columns
        assert self.model is not None
        assert self.model.is_fitted

        catalog = self.item2idx.external_ids

        hot_pairs = (
            pairs_df
            .filter(pl.col("user_id").is_in(set(self.user2idx.external_ids)))
            .filter(pl.col("item_id").is_in(set(catalog)))
        )

        assert catalog.shape[0] == self.item_embeddings.shape[0]

        user2item_dist = (
            self.user_embeddings[self.user2idx.convert_to_internal(hot_pairs["user_id"].to_numpy())]
            * self.item_embeddings[self.item2idx.convert_to_internal(hot_pairs["item_id"].to_numpy())]
        ).sum(axis=1)

        
        hot_pairs = (
            hot_pairs
            .with_columns(
                pl.lit(user2item_dist).cast(pl.Float32).alias(self.predict_col_name)
            )
        )

        cold_pairs = (
            pairs_df
            .join(hot_pairs.select("user_id", "item_id"), how="anti", on=["user_id", "item_id"])
            .with_columns(
                pl.lit(self.cold_predict).cast(pl.Float32).alias(self.predict_col_name),
            )
        )

        logger.info(f"ALS: Percent of cold pairs: {float(cold_pairs.shape[0]) / pairs_df.shape[0]}")

        return pl.concat([hot_pairs, cold_pairs])
