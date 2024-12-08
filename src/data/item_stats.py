import polars as pl

from src.logger import logger


def get_item_stats(
    train_df: pl.DataFrame,
    items_meta_df: pl.DataFrame,
    users_meta_df: pl.DataFrame,
    column: str = "item_id",  # source_id
    positive_threshold_for_ratio: int = 10,
    ratio_default_value: float = 1.0,
    min_users_for_stats: int = 20,
    max_timespent_ratio: float = 5.0
) -> pl.DataFrame:
    item_features_extra = (
        train_df
        .drop(["rn", "max_rn"])
        .join(items_meta_df.select("item_id", "source_id", "duration"), how="left", on="item_id")
        .join(users_meta_df, how="left", on="user_id")
        .with_columns(
            (pl.col("timespent") / pl.col("duration")).clip(0, max_timespent_ratio).alias("timespent_ratio"),
            (pl.col("like") + pl.col("share") + pl.col("bookmarks")).alias("positive_feedback")
        )
        .group_by(column)
        .agg(
            pl.first().count().alias("n_users"),
            pl.col("like").mean().alias("mean_like"),
            pl.col("like").sum().alias("sum_like"),
            pl.col("dislike").mean().alias("mean_dislike"),
            pl.col("dislike").sum().alias("sum_dislike"),
            pl.col("share").mean().alias("mean_share"),
            pl.col("share").sum().alias("sum_share"),
            pl.col("bookmarks").mean().alias("mean_bookmarks"),
            pl.col("bookmarks").sum().alias("sum_bookmarks"),
            pl.col("timespent_ratio").mean().alias("mean_timespent_ratio"),
            pl.col("gender").mean().alias("mean_gender"),
            pl.col("gender").filter(pl.col("positive_feedback") >= 1).mean().alias("mean_gender_positive"),
            pl.col("gender").filter(pl.col("dislike") >= 1).mean().alias("mean_gender_negative"),
            pl.col("age").mean().alias("mean_age"),
            pl.col("age").filter(pl.col("positive_feedback") >= 1).mean().alias("mean_age_positive"),
            pl.col("age").filter(pl.col("dislike") >= 1).mean().alias("mean_age_negative"),
            pl.col("positive_feedback").sum().alias("sum_positive"),
        )
        .filter(pl.col("n_users") >= min_users_for_stats)
        .with_columns(
            pl.when(pl.col("sum_positive") >= positive_threshold_for_ratio).then(
                ((pl.col("sum_positive") - pl.col("sum_dislike")) / (pl.col("sum_positive") + pl.col("sum_dislike")))
            ).otherwise(None).alias("like_dislike_ratio"),
            pl.when(pl.col("sum_positive") >= positive_threshold_for_ratio).then("mean_gender_positive").otherwise(None).alias("mean_gender_positive"),
            pl.when(pl.col("sum_positive") >= positive_threshold_for_ratio).then("mean_age_negative").otherwise(None).alias("mean_age_negative"),
        )
        .with_columns(
            pl.col("like_dislike_ratio").fill_null(ratio_default_value)
        )
        .drop(
            "n_users",
            # "sum_like",
            # "sum_dislike",
            # "sum_share",
            # "sum_bookmarks",
            # "sum_positive"
        )
    )

    return item_features_extra


def get_user2source_stats(
    df: pl.DataFrame, 
    items_meta_df: pl.DataFrame,
    min_interactions_threshold: int = 10,
) -> pl.DataFrame:
    logger.debug("Calculate train interactions")
    total_interactions = df.group_by("user_id").len().select("user_id", pl.col("len").alias("n_total_interactions"))
    logger.debug("Done calculate train interactions")

    logger.debug("Calculate user 2 source stats")
    result = (
        df.lazy()
        .join(items_meta_df.lazy(), on="item_id", how="inner")
        .with_columns(((pl.col("like") + pl.col("share") + pl.col("bookmarks")) > 0).alias("is_positive"))
        .group_by("user_id", "source_id")
        .agg(
            pl.first().len().alias("n_interactions"),
            pl.col("is_positive").cast(int).mean().alias("source_acceptance"),
            pl.col("dislike").cast(int).mean().alias("source_disacceptance"),
            (pl.col("timespent") / pl.col("duration")).mean().alias("mean_source_timespent_ratio")
        )
        .join(total_interactions.lazy(), on="user_id", how="inner")
        .with_columns(
            pl.col("n_interactions").truediv("n_total_interactions").alias("source_perc")
        )
        .filter(pl.col("n_interactions") >= min_interactions_threshold)
        .drop("n_interactions", "n_total_interactions")
        .collect()
    )

    logger.debug("Done calculate user 2 source stats")

    return result
