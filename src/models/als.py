from pathlib import Path

import polars as pl
from implicit.als import AlternatingLeastSquares
from rectools.models import ImplicitALSWrapperModel
from rectools.dataset import Dataset

from src.logger import logger
from src.models.base_matrix_factorization import BaseMatrixFactorization


class ALSModel(BaseMatrixFactorization):
    def __init__(
        self,
        iterations: int = 10,
        alpha: float = 10,
        regularization: float = 0.01,
        n_factors: int = 64,
        fit_features_together: bool = False,
        use_gpu: bool = False,
        num_threads: int = 8,
        random_state: int = 42,
        predict_col_name: str = "predict",
        cold_predict: float = -1.0,
        cache_dir: Path | None = None
    ):
        self.iterations = iterations
        self.alpha = alpha
        self.regularization = regularization
        self.n_factors = n_factors
        self.fit_features_together = fit_features_together
        self.use_gpu = use_gpu
        self.num_threads = num_threads
        self.random_state = random_state
        self.predict_col_name = predict_col_name
        self.cold_predict = cold_predict

        super().__init__(
            cache_dir=cache_dir, 
            iterations=iterations,
            alpha=alpha,
            regularization=regularization,
            n_factors=n_factors,
            fit_features_together=fit_features_together,
            use_gpu=use_gpu,
            num_threads=num_threads,
            random_state=random_state,
            predict_col_name=predict_col_name,
            cold_predict=cold_predict,
        )

        self.model = ImplicitALSWrapperModel(
            AlternatingLeastSquares(
                factors=self.n_factors,
                regularization=self.regularization,
                alpha=self.alpha,
                random_state=self.random_state,
                use_gpu=self.use_gpu,
                num_threads=self.num_threads,
                iterations=iterations
            ),
            fit_features_together=self.fit_features_together,
        )

        self.user2idx = None
        self.nmid2idx = None

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

        additional_params = {}

        if items_meta_df_flatten is not None:
            additional_params.update({
                "item_features_df": items_meta_df_flatten.to_pandas(),
                "cat_item_features": items_meta_df_flatten["feature"].unique().to_list(),
            })

        if users_meta_df_flatten is not None:
            additional_params.update({
                "user_features_df": users_meta_df_flatten.to_pandas(),
                "cat_user_features": users_meta_df_flatten["feature"].unique().to_list(),
            })
        
        dataset = Dataset.construct(
            interactions_df=train_df.to_pandas(),
            **additional_params
        )

        self.user2idx = dataset.user_id_map
        self.item2idx = dataset.item_id_map

        self.model.fit(dataset)

        self.is_fitted = self.model.is_fitted

        self._save_cache(train_df, items_meta_df_flatten, users_meta_df_flatten)

        return self
    
    def predict_proba(self, pairs_df: pl.DataFrame) -> pl.DataFrame:
        assert self.is_fitted
        assert self.user2idx
        assert self.item2idx
        assert "user_id" in pairs_df.columns
        assert "item_id" in pairs_df.columns

        catalog = self.item2idx.external_ids

        hot_pairs = (
            pairs_df
            .filter(pl.col("user_id").is_in(set(self.user2idx.external_ids)))
            .filter(pl.col("item_id").is_in(set(catalog)))
        )

        user_embeddings, item_embeddings = self.model.get_vectors()

        assert catalog.shape[0] == item_embeddings.shape[0], f"Shape mismatch {catalog.shape[0]} != {item_embeddings.shape[0]}"

        user2item_dist = (
            user_embeddings[self.user2idx.convert_to_internal(hot_pairs["user_id"].to_numpy())]
            * item_embeddings[self.item2idx.convert_to_internal(hot_pairs["item_id"].to_numpy())]
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
                pl.lit(self.cold_predict).cast(pl.Float32).alias(self.predict_col_name)
            )
        )

        logger.info(f"ALS: Percent of cold pairs: {float(cold_pairs.shape[0]) / pairs_df.shape[0]}")

        return pl.concat([hot_pairs, cold_pairs])
