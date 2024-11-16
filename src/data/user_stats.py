import polars as pl


def get_user_stats(
    df: pl.DataFrame,
    items_meta_df: pl.DataFrame,
    min_items_for_stats: int = 10,
    max_timespent_ratio: float = 5.0
) -> pl.DataFrame:
    feature_columns = (
        "avg_timespent", 
        "avg_timespent_like", 
        "avg_timespent_dislike", 
        "avg_timespent_ratio", 
        "avg_timespent_ratio_like", 
        "avg_timespent_ratio_dislike", 
        # "total_interactions", 
        "like_perc", 
        "dislike_perc", 
        "share_perc", 
        "bookmarks_perc",
    )

    return (
        df
        .join(items_meta_df.select("item_id", "duration"), how="left", on="item_id")
        .with_columns((pl.col("timespent") / pl.col("duration")).clip(0, max_timespent_ratio).alias("timespent_ratio"))
        .group_by("user_id")
        .agg(
            pl.col("like").sum().alias("total_likes"),
            pl.col("dislike").sum().alias("total_dislikes"),
            pl.col("share").sum().alias("total_share"),
            pl.col("bookmarks").sum().alias("total_bookmarks"),
            pl.col("timespent").mean().alias("avg_timespent"),
            pl.col("timespent").filter(pl.col("like") == 1).mean().alias("avg_timespent_like"),
            pl.col("timespent").filter(pl.col("dislike") == 1).mean().alias("avg_timespent_dislike"),
            pl.col("timespent_ratio").mean().alias("avg_timespent_ratio"),
            pl.col("timespent_ratio").filter(pl.col("like") == 1).mean().alias("avg_timespent_ratio_like"),
            pl.col("timespent_ratio").filter(pl.col("dislike") == 1).mean().alias("avg_timespent_ratio_dislike"),
            pl.count().alias("total_interactions"),
        )
        .filter(pl.col("total_interactions") >= min_items_for_stats)
        .with_columns(
            (pl.col("total_likes") / pl.col("total_interactions")).alias("like_perc"),
            (pl.col("total_dislikes") / pl.col("total_interactions")).alias("dislike_perc"),
            (pl.col("total_share") / pl.col("total_interactions")).alias("share_perc"),
            (pl.col("total_bookmarks") / pl.col("total_interactions")).alias("bookmarks_perc"),
        )
        .select(
            "user_id",
            *[pl.col(col).alias(f"user_{col}") for col in feature_columns]
        )
    )
