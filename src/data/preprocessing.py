from pathlib import Path

import polars as pl
from rectools import Columns

from src.logger import logger


def load_data(data_dir: Path = Path("../data/processed/")) -> dict[str, pl.DataFrame]:
    logger.debug("Load data")
    data = dict(
        train_df_als=pl.read_parquet(data_dir / "train_df_als.parquet"),
        train_df_cb=pl.read_parquet(data_dir / "train_df_cb.parquet"),
        test_df=pl.read_parquet(data_dir / "test_df.parquet"),
    )
    logger.debug("Done loading data")

    return data


def prepare_train_for_als_item_like(df: pl.DataFrame) -> pl.DataFrame:
    logger.debug(f"Preprocessing data for item als. Input size: {df.shape[0]}")

    train_df_als_like = (
        df
        .filter((pl.col("like") + pl.col("dislike")) >= 1)
        .with_columns(weight=pl.col("like").cast(pl.Int8) - pl.col("dislike").cast(pl.Int8))
        .select("user_id", "item_id", pl.col("weight").alias(Columns.Weight), pl.lit(1).alias(Columns.Datetime))
    )

    logger.debug(f"Output shape: {tuple(train_df_als_like.shape)}")

    return train_df_als_like


def prepare_train_for_als_item_like_book_share(
    df: pl.DataFrame,
    share_weight: float = 1.0,
    bookmarks_weight: float = 1.0,
    like_weight: float = 1.0,
) -> pl.DataFrame:
    logger.debug(f"Preprocessing data for item als. Input size: {df.shape[0]}")

    train_df_als_like_book_share = (
        df
        .filter((pl.col("like") + pl.col("dislike") + pl.col("share") + pl.col("bookmarks")) >= 1)
        .with_columns(
            weight=(
                pl.when(pl.col("dislike") == 0).then(
                    like_weight * pl.col("like")
                    + share_weight * pl.col("share")
                    + bookmarks_weight * pl.col("bookmarks")
                ).otherwise(-1).cast(pl.Int8)
            )
        )
        .select("user_id", "item_id", pl.col("weight").alias(Columns.Weight), pl.lit(1).alias(Columns.Datetime))
    )

    logger.debug(f"Output shape: {tuple(train_df_als_like_book_share.shape)}")

    return train_df_als_like_book_share


def prepare_train_for_als_source_id(df_for_als: pl.DataFrame, items_meta_df: pl.DataFrame) -> pl.DataFrame:
    logger.debug(f"Preprocessing data for source als. Input size: {df_for_als.shape[0]}")

    df_source_id = (
        df_for_als
        .join(items_meta_df.select("item_id", "source_id"), how="inner", on="item_id")
        .drop("item_id")
        .rename({"source_id": "item_id"})
    )

    logger.debug(f"Output shape: {tuple(df_source_id.shape)}")

    return df_source_id


def prepare_train_for_als_timespent(df: pl.DataFrame, items_meta_df: pl.DataFrame) -> pl.DataFrame:
    logger.debug(f"Preprocessing data for item als on timespent. Input size: {df.shape[0]}")

    df_als_timespent_lazy = (
        df
        .lazy()
        .join(items_meta_df.select("item_id", "duration").lazy(), on="item_id", how="inner")
        .with_columns((pl.col("timespent") / pl.col("duration")).clip(0, 3).alias("timespent_ratio"))
        .filter(pl.col("timespent_ratio") > 0.5)
        .select(
            "user_id", 
            "item_id", 
            pl.col("timespent_ratio").alias("weight"), 
            pl.col("rn").alias("datetime")
        )
        .collect()
    )

    logger.debug(f"Output shape: {tuple(df_als_timespent_lazy.shape)}")

    return df_als_timespent_lazy