from pathlib import Path
import polars as pl

from src.models.als import ALSModel
from src.logger import logger


class ALSSource(ALSModel):
    def __init__(
        self, 
        items_meta_df: pl.DataFrame,
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
        cache_dir: Path | None = None,
    ):
        super().__init__(
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
            cache_dir=cache_dir,
        )

        self.items_meta_df = items_meta_df

    def fit(
        self, 
        train_df: pl.DataFrame, 
        items_meta_df_flatten: pl.DataFrame | None = None, 
        users_meta_df_flatten: pl.DataFrame | None = None
    ):
        if (items_meta_df_flatten is not None) or (users_meta_df_flatten is not None):
            logger.warning("items_meta_df_flatten or users_meta_df_flatten is not None. However it is not used in this model")
        
        train_df = (
            train_df
            .join(self.items_meta_df.select("item_id", "source_id"), how="inner", on="item_id")
            .drop("item_id")
            .rename({"source_id": "item_id"})
        )
        
        return super().fit(train_df, None, None)

    def predict_proba(self, pairs_df: pl.DataFrame) -> pl.DataFrame:
        logger.debug(f"Input df columns: {pairs_df.columns}")
        pairs_source_id = (
            pairs_df
            .join(self.items_meta_df.select("item_id", "source_id"), how="left", on="item_id")
        )

        uniq_source_pairs_df = (
            pairs_source_id
            .drop_nulls(["user_id", "source_id"])
            .unique(["user_id", "source_id"])
            .drop("item_id")
            .rename({"source_id": "item_id"})
        )
        logger.debug(f"uniq_source_pairs_df shape: {uniq_source_pairs_df.shape}; columns: {uniq_source_pairs_df.columns}")

        # [user_id, item_id-по факту source_id]
        uniq_predict = super().predict_proba(uniq_source_pairs_df).rename({"item_id": "source_id"})
        logger.debug(f"uniq_predict shape: {uniq_predict.shape}; columns: {uniq_predict.columns}")

        predict = (
            pairs_source_id
            .select("user_id", "item_id", "source_id")
            .join(uniq_predict, how="inner", on=["user_id", "source_id"])
            .drop("source_id")
            .with_columns(
                pl.col(self.predict_col_name).fill_null(self.cold_predict)
            )
        )
        logger.debug(f"predict shape: {predict.shape}; columns: {predict.columns}")

        return predict
