import polars as pl

from src.logger import logger


def get_user_stats(
    df: pl.DataFrame,
    items_meta_df: pl.DataFrame,
    min_items_for_stats: int = 10,
    max_timespent_ratio: float = 5.0
) -> pl.DataFrame:
    logger.debug("Getting user stats")
    df_result = (
        df
        .join(items_meta_df.select("item_id", "source_id", "duration"), how="left", on="item_id")
        .with_columns(
            (pl.col("timespent") / pl.col("duration")).clip(0, max_timespent_ratio).alias("timespent_ratio"),
            pl.col("rn").cum_count(reverse=True).over("user_id").alias("rn_new")
        )
        .group_by("user_id")
        .agg(
            pl.col("source_id").mode().alias("source_id_mode"),
            pl.col("source_id").filter(pl.col("like") == 1).mode().alias("liked_source_id_mode"),
            pl.col("like").sum().alias("total_likes"),
            pl.col("dislike").sum().alias("total_dislikes"),
            pl.col("share").sum().alias("total_share"),
            pl.col("bookmarks").sum().alias("total_bookmarks"),
            pl.col("timespent").sum().alias("total_timespent"),
            pl.col("timespent").mean().alias("avg_timespent"),
            *[
                pl.col("timespent").quantile(i / 10.0).alias(f"timespent_q{i * 10}")
                for i in range(1, 10, 2)
            ],
            pl.col("timespent").filter(pl.col("like") == 1).mean().alias("avg_timespent_like"),
            pl.col("timespent").filter(pl.col("dislike") == 1).mean().alias("avg_timespent_dislike"),
            pl.col("timespent_ratio").mean().alias("avg_timespent_ratio"),
            pl.col("timespent_ratio").filter(pl.col("like") == 1).mean().alias("avg_timespent_ratio_like"),
            pl.col("timespent_ratio").filter(pl.col("dislike") == 1).mean().alias("avg_timespent_ratio_dislike"),
            pl.count().alias("total_interactions"),
            # like
            pl.col("like").filter(pl.col("rn_new") <= 10).sum().alias("n_likes_last_10"),
            pl.col("like").filter(pl.col("rn_new") <= 20).sum().alias("n_likes_last_20"),
            pl.col("like").filter(pl.col("rn_new") <= 100).sum().alias("n_likes_last_100"),
            # dislike
            pl.col("dislike").filter(pl.col("rn_new") <= 10).sum().alias("n_dislikes_last_10"),
            pl.col("dislike").filter(pl.col("rn_new") <= 20).sum().alias("n_dislikes_last_20"),
            pl.col("dislike").filter(pl.col("rn_new") <= 100).sum().alias("n_dislikes_last_100"),
            # share
            pl.col("share").filter(pl.col("rn_new") <= 10).sum().alias("n_shares_last_10"),
            pl.col("share").filter(pl.col("rn_new") <= 20).sum().alias("n_shares_last_20"),
            pl.col("share").filter(pl.col("rn_new") <= 100).sum().alias("n_shares_last_100"),
            # bookmarks
            pl.col("bookmarks").filter(pl.col("rn_new") <= 10).sum().alias("n_bookmarks_last_10"),
            pl.col("bookmarks").filter(pl.col("rn_new") <= 20).sum().alias("n_bookmarks_last_20"),
            pl.col("bookmarks").filter(pl.col("rn_new") <= 100).sum().alias("n_bookmarks_last_100"),
        )
        .filter(pl.col("total_interactions") >= min_items_for_stats)
        .with_columns(
            (pl.col("total_likes") / pl.col("total_interactions")).alias("like_perc"),
            (pl.col("total_dislikes") / pl.col("total_interactions")).alias("dislike_perc"),
            (pl.col("total_share") / pl.col("total_interactions")).alias("share_perc"),
            (pl.col("total_bookmarks") / pl.col("total_interactions")).alias("bookmarks_perc"),
        )
    )

    feature_columns = [c for c in df_result.columns if c != "user_id"]
    return df_result.select(
        "user_id",
        *[pl.col(col).alias(f"user_{col}") for col in feature_columns]
    )
