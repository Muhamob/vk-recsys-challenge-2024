import polars as pl


def calc_user_auc(df: pl.DataFrame, predict_col: str = "predictions", target_col: str = "target") -> pl.DataFrame:
    """
    Поюзерный rocauc с учётом нескольких градаций таргета (like/no reaction/dislike)
    """
    dd1 = (
        df
        .filter(pl.col(target_col) == 1)
        .join(df.filter(pl.col(target_col) == 0), how="inner", on=["user_id"], suffix="_r")
        .group_by("user_id")
        .agg(
            pl.when(pl.col(predict_col) > pl.col(f"{predict_col}_r")).then(1)
            .when(pl.col(predict_col) == pl.col(f"{predict_col}_r")).then(0.5)
            .otherwise(0).sum().alias("num"),
            pl.count().alias("den")
        )
    )
    
    dd2 = (
        df
        .filter(pl.col(target_col) == 0)
        .join(df.filter(pl.col(target_col) == -1), how="inner", on=["user_id"], suffix="_r")
        .group_by("user_id")
        .agg(
            pl.when(pl.col(predict_col) > pl.col(f"{predict_col}_r")).then(1)
            .when(pl.col(predict_col) == pl.col(f"{predict_col}_r")).then(0.5)
            .otherwise(0).sum().alias("num"),
            pl.count().alias("den")
        )
    )

    return (
        dd1
        .join(dd2, how="full", on="user_id", suffix="_d")
        .with_columns(
            pl.coalesce("user_id", "user_id_d").alias("user_id"),
        )
        .fill_null(0)
        .with_columns(
            (pl.col("num_d") + pl.col("num")).alias("num"),
            (pl.col("den_d") + pl.col("den")).alias("den"),
        )
        .with_columns(
            (pl.col("num") / pl.col("den")).alias("rocauc")
        )
        .drop("user_id_d", "num_d", "den_d")
    )
