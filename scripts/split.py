from pathlib import Path

import polars as pl
import click


@click.command()
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False), default="./data/raw")
@click.option("--save-dir", type=click.Path(exists=True, file_okay=False), default="./data/processed")
@click.option("--test-n-items", type=click.INT, default=20)
@click.option("--train-for-cb-size", type=click.FLOAT, default=0.2)
def split_data(
    data_dir: Path,
    save_dir: Path,
    test_n_items: int,
    train_for_cb_size: float,
):
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    
    interactions_df = pl.read_parquet(data_dir / "train_interactions.parquet")

    test_df = (
        interactions_df
        .group_by("user_id")
        .agg(pl.col("item_id").count().alias("n_items"))
        .filter(pl.col("n_items") > test_n_items)
        .drop("n_items")
        .join(interactions_df, how="left", on="user_id")
        .with_columns(pl.first().cum_count(reverse=True).alias("row_number").over("user_id").alias("rn"))
        .filter(pl.col("rn") <= test_n_items)
        .drop("rn")
    )

    train_df = (
        interactions_df
        .join(test_df.select("user_id", "item_id"), how="anti", on=["user_id", "item_id"])
    )

    train_df_stats = (
        train_df
        .lazy()
        .with_columns(
            pl.first().cum_count(reverse=True).over("user_id").alias("rn"),
            pl.first().count().over("user_id").alias("max_rn"),
        )
        .with_columns(
            (pl.col("rn") / pl.col("max_rn")).alias("rn_ratio")
        )
    )

    train_df_cb = (
        train_df_stats
        .filter(
            pl.col("rn_ratio") < train_for_cb_size
        )
        .drop("rn_ratio")
        .collect()
    )

    train_df_als = (
        train_df_stats
        .filter(
            pl.col("rn_ratio") >= train_for_cb_size
        )
        .drop("rn_ratio")
        .collect()
    )

    train_df_als.write_parquet(save_dir / "train_df_als.parquet")
    train_df_cb.write_parquet(save_dir / "train_df_cb.parquet")
    test_df.write_parquet(save_dir / "test_df.parquet")


if __name__ == "__main__":
    split_data()
