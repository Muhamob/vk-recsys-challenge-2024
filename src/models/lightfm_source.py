from pathlib import Path
import polars as pl

from src.models.lightfm import LFMModel
from src.logger import logger


class LightFMSourceAdd(LFMModel):
    def __init__(
        self, 
        items_meta_df: pl.DataFrame,
        n_features: int = 10,
        n_epochs: int = 10,
        loss: str = "bpr",
        predict_col_name: str = "predict",
        cold_predict: float = -1.0,
        verbose: int = 0,
        random_state: int = 42,
        cache_dir: Path | None = None,
    ):
        super().__init__(
            n_features=n_features,
            n_epochs=n_epochs,
            loss=loss,
            predict_col_name=predict_col_name,
            cold_predict=cold_predict,
            verbose=verbose,
            random_state=random_state,
            cache_dir=cache_dir,
        )

        self.items_meta_df = items_meta_df.select("item_id", "source_id")

    def fit(
        self, 
        train_df: pl.DataFrame, 
        items_meta_df_flatten: pl.DataFrame | None = None, 
        users_meta_df_flatten: pl.DataFrame | None = None
    ):
        if (items_meta_df_flatten is not None) or (users_meta_df_flatten is not None):
            logger.warning("items_meta_df_flatten or users_meta_df_flatten is not None. However it is not used in this model")

        train_df = train_df.with_columns(pl.col("weight").cast(pl.Float32))

        train_df = pl.concat([
            train_df,
            (
                train_df
                .join(self.items_meta_df.select("item_id", "source_id"), on="item_id", how="inner")
                .drop("item_id")
                .group_by("user_id", pl.col("source_id").alias("item_id") + 1_000_000)
                .agg(pl.col("weight").sum(), pl.max("datetime"))
            )
        ])
        
        return super().fit(train_df, None, None)


class LightFMSource(LFMModel):
    def __init__(
        self, 
        items_meta_df: pl.DataFrame,
        n_features: int = 10,
        n_epochs: int = 10,
        loss: str = "bpr",
        predict_col_name: str = "predict",
        cold_predict: float = -1.0,
        verbose: int = 0,
        random_state: int = 42,
        cache_dir: Path | None = None,
    ):
        super().__init__(
            n_features=n_features,
            n_epochs=n_epochs,
            loss=loss,
            predict_col_name=predict_col_name,
            cold_predict=cold_predict,
            verbose=verbose,
            random_state=random_state,
            cache_dir=cache_dir,
        )

        self.items_meta_df = items_meta_df.select("item_id", "source_id")

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
            .join(self.items_meta_df, how="inner", on="item_id")
            .drop("item_id")
            .rename({"source_id": "item_id"})
        )
        
        return super().fit(train_df, None, None)

    def predict_proba(self, pairs_df: pl.DataFrame) -> pl.DataFrame:
        logger.debug(f"Input df columns: {pairs_df.columns}")
        pairs_source_id = (
            pairs_df
            .join(self.items_meta_df, how="left", on="item_id")
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
            # .with_columns(
            #     pl.col(self.predict_col_name).fill_null(self.cold_predict)
            # )
        )
        logger.debug(f"predict shape: {predict.shape}; columns: {predict.columns}")

        return predict
