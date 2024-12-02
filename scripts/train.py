from datetime import datetime
import gc
from itertools import product
import os
from pathlib import Path
from typing import Sequence
import threadpoolctl
import logging

import polars as pl
import numpy as np
from tqdm import tqdm
from catboost import CatBoostClassifier, CatBoostRanker, Pool
import click
import mlflow
import pandas as pd

from src.data.item_stats import get_item_stats, get_user2source_stats
from src.data.user_stats import get_user_stats
from src.metrics import calc_user_auc
from src.models.als import ALSModel
from src.models.als_source import ALSSource
from src.models.lightfm import LFMModel
from src.models.lightfm_source import LightFMSource, LightFMSourceAdd
from src.models.ease import EASEModel, EASESourceModel
from src.data.preprocessing import (
    add_log_weight, 
    load_data, 
    prepare_train_for_als_item_like,
    prepare_train_for_als_item_like_all, 
    prepare_train_for_als_item_like_book_share, 
    prepare_train_for_als_timespent,
    train_test_split_by_user_id,
)
from src.logger import logger


os.environ["OPENBLAS_NUM_THREADS"] = "1"
threadpoolctl.threadpool_limits(1, "blas")


def calc_mean_embedding(embeddings: pl.Series):
    return embeddings.to_numpy().mean(axis=0).tolist()


def dot_product(x):
    left_embeddings = x.struct[0].to_numpy()
    right_embeddings = x.struct[1].to_numpy()

    dot_product = (left_embeddings * right_embeddings).sum(axis=1)

    return dot_product


def get_emb_sim_features(df_pairs, items_meta_df, user_liked_mean_embeddings, user_disliked_mean_embeddings):
    chunk_size = 100_000
    sliced_result = []
    for df_sample in tqdm(df_pairs.select("user_id", "item_id").iter_slices(n_rows=chunk_size), total=df_pairs.shape[0] // chunk_size + 1):
        df_sample = (
            df_sample
            .join(items_meta_df.select("item_id", "embeddings"), on="item_id", how="inner")
        )

        liked_sim = (
            df_sample
            .join(user_liked_mean_embeddings, on="user_id", how="inner")
            .with_columns(pl.struct("embeddings", "embeddings_right").map_batches(dot_product).alias("liked_embeddings_sim"))
            .select("user_id", "item_id", "liked_embeddings_sim")
        )
    
        disliked_sim = (
            df_sample
            .join(user_disliked_mean_embeddings, on="user_id", how="inner")
            .with_columns(pl.struct("embeddings", "embeddings_right").map_batches(dot_product).alias("disliked_embeddings_sim"))
            .select("user_id", "item_id", "disliked_embeddings_sim")
        )
    
        sliced_result.append((
            df_sample
            .select("user_id", "item_id")
            .join(liked_sim, how="left", on=["user_id", "item_id"])
            .join(disliked_sim, how="left", on=["user_id", "item_id"])
        ))
    
    return (
        pl.concat(sliced_result)
        .with_columns(
            pl.col("liked_embeddings_sim").fill_null(pl.col("liked_embeddings_sim").mean()),
            pl.col("disliked_embeddings_sim").fill_null(pl.col("disliked_embeddings_sim").mean())
        )
    )


def join_features(
    target: pl.DataFrame, 
    predict: pl.DataFrame, 
    item_stats: pl.DataFrame, 
    source_stats: pl.DataFrame, 
    items_meta_df: pl.DataFrame,
    sim_features: pl.DataFrame | None,
    users_meta_df: pl.DataFrame | None,
    user_stats: pl.DataFrame,
    user2source_stats: pl.DataFrame,  # [user_id, source_id, features...]
):
    df = (
        target
        .join(predict, how="left", on=["user_id", "item_id"])
        .join(item_stats, how="left", on=["item_id"], suffix="_item")
        .join(items_meta_df.select("item_id", "source_id"), how="left", on="item_id")
        .join(source_stats, how="left", on="source_id", suffix="_source")
        .join(user2source_stats, on=["user_id", "source_id"], how="left", suffix="_user2source_feature")
        .join(user_stats, how="left", on="user_id")
    )

    if sim_features is not None:
        df = df.join(sim_features, how="left", on=["user_id", "item_id"])

    if users_meta_df is not None:
        df = df.join(users_meta_df, how="left", on="user_id")

        age_features = [col for col in df.columns if "age" in col]
        age_features = [col for col in age_features if col != "age"]
        df = (
            df
            .with_columns(*[
                (pl.col("age") - pl.col(col)).alias(f"{col}_diff")
                for col in age_features
            ])
            .with_columns(*[
                (pl.col("age") - pl.col(col)).abs().alias(f"{col}_diff_abs")
                for col in age_features
            ])
        )
        print([col for col in df.columns if "age" in col])
        
        gender_features = [col for col in df.columns if "gender" in col]
        gender_features = [col for col in gender_features if col != "gender"]
        df = (
            df
            .with_columns(*[
                (pl.col("gender") - pl.col(col)).alias(f"{col}_diff")
                for col in gender_features
            ])
            .with_columns(*[
                (pl.col("gender") - pl.col(col)).abs().alias(f"{col}_diff_abs")
                for col in gender_features
            ])
        )
        print([col for col in df.columns if "gender" in col])

    # extra features
    df = (
        df
        .with_columns(
            (pl.col("user_like_perc") * pl.col("mean_like")).alias("user2item_like_perc"),
            (
                pl.col("mean_gender").sub(1).pow(pl.col("gender").sub(1))
                * pl.col("mean_gender").sub(2).mul(-1).pow(pl.col("gender").sub(2).mul(-1))
            ).alias("gender_distr"),
            (
                pl.col("mean_gender_source").sub(1).pow(pl.col("gender").sub(1))
                * pl.col("mean_gender_source").sub(2).mul(-1).pow(pl.col("gender").sub(2).mul(-1))
            ).alias("gender_source_distr"),
        )
        .with_columns(
            pl.col("user2item_like_perc").mul("gender_distr").alias("user2item_like_perc_gender")
        )
    )

    return (
        df
        .drop("source_id")
        .sort("user_id", "item_id")
    )


def add_poly_features(df: pl.DataFrame, feature_columns: Sequence[str]) -> tuple[pl.DataFrame, Sequence[str]]:
    return (df, feature_columns)


def get_cb_pool(df, feature_columns, add_labels: bool = True) -> Pool:
    return Pool(
        data=df[feature_columns],
        label=df["target"] if add_labels else None,
        group_id=df["user_id"],
        embedding_features=["embeddings", ]
    )


@click.group()
@click.option("--level", default="info")
def cli(level):
    logger.setLevel({
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "error": logging.ERROR,
    }[level])



@cli.command()
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False), default="./data/")
@click.option("--save-datasets", type=click.BOOL, default=False)
def train(data_dir: Path, save_datasets: bool):
    # PYTHONPATH=. python scripts/train.py --level debug train --data-dir ./data
    mlflow.set_experiment("vk")
    mlflow.set_tracking_uri("http://127.0.0.1:8080") 

    with mlflow.start_run():
        data_dir = Path(data_dir)

        datasets = load_data(data_dir / "processed")
        test_pairs = pl.read_csv(data_dir / "raw/test_pairs.csv.csv").with_columns(
            pl.col("item_id").cast(pl.UInt32),
            pl.col("user_id").cast(pl.UInt32)
        )
        items_meta_df = pl.read_parquet(data_dir / "raw/items_meta.parquet")
        users_meta_df = pl.read_parquet(data_dir / "raw/users_meta.parquet")

        # filter users with no target
        for ds in ["train_df_cb", "test_df"]:
            logger.info(f"{ds} rows before filter: {datasets[ds].shape[0]}")

            datasets[ds] = (
                datasets[ds]
                .group_by("user_id")
                .agg(
                    pl.col("item_id").count().alias("n_items"),
                    pl.col("like").sum().alias("n_likes"),
                    pl.col("dislike").sum().alias("n_dislikes"),
                    pl.col("share").sum().alias("n_share"),
                    pl.col("bookmarks").sum().alias("n_bookmarks"),
                )
                .with_columns(any_target_sum=pl.col("n_likes") + pl.col("n_dislikes") + pl.col("n_share") + pl.col("n_bookmarks"))
                .filter(pl.col("any_target_sum") > 0)
                .select("user_id")
                .join(datasets[ds], how="left", on="user_id")
            )

            logger.info(f"{ds} rows after filter: {datasets[ds].shape[0]}")

        like_weight_item_like = 1
        train_als_like_item = prepare_train_for_als_item_like(datasets["train_df_als"], like_weight=like_weight_item_like)
        train_als_like_book_share_item = prepare_train_for_als_item_like_book_share(datasets["train_df_als"], like_weight=like_weight_item_like)
        train_als_timespent = prepare_train_for_als_timespent(datasets["train_df_als"], items_meta_df=items_meta_df)
        train_als_like_item_all = prepare_train_for_als_item_like_all(datasets["train_df_als"])

        train_als_like_item_time_weighted = add_log_weight(train_als_like_item)
        train_als_like_book_share_item_time_weighted = add_log_weight(train_als_like_book_share_item)

        item_stats = get_item_stats(datasets["train_df_als"], items_meta_df, users_meta_df, column="item_id")
        source_stats = get_item_stats(datasets["train_df_als"], items_meta_df, users_meta_df, column="source_id")
        user_stats = get_user_stats(datasets["train_df_als"], items_meta_df=items_meta_df)

        user_liked_mean_embeddings = (
            datasets["train_df_als"]
            .filter(pl.col("like") + pl.col("share") + pl.col("bookmarks") >= 1)
            .join(items_meta_df.select("item_id", "source_id", "embeddings"), how="left", on="item_id")
            .group_by("user_id")
            .agg(pl.col("embeddings").map_elements(calc_mean_embedding))
            .with_columns(pl.col("embeddings").list.to_array(32))
        )

        user_disliked_mean_embeddings = (
            datasets["train_df_als"]
            .filter(pl.col("dislike") == 1)
            .join(items_meta_df.select("item_id", "source_id", "embeddings"), how="left", on="item_id")
            .group_by("user_id")
            .agg(pl.col("embeddings").map_elements(calc_mean_embedding))
            .with_columns(pl.col("embeddings").list.to_array(32))
        )

        train_df_cb_sim_features = get_emb_sim_features(datasets["train_df_cb"], items_meta_df, user_liked_mean_embeddings, user_disliked_mean_embeddings)
        test_df_sim_features = get_emb_sim_features(datasets["train_df_cb"], items_meta_df, user_liked_mean_embeddings, user_disliked_mean_embeddings)
        test_pairs_sim_features = get_emb_sim_features(test_pairs, items_meta_df, user_liked_mean_embeddings, user_disliked_mean_embeddings)

        user2source_stats = get_user2source_stats(datasets["train_df_als"], items_meta_df)

        # models
        ## ALS
        iterations = 49
        alpha = 44
        regularization = 0.07702668794141683
        # n_factors = 96
        n_factors = 128

        # lfm
        lfm_n_features = 128
        lfm_n_epochs = 50
        lfm_source_add_n_epochs = 50

        # ease
        ease_max_items = 35_000
        ease_max_source_ids = 5000
        ease_regularization = 2000.0

        als_cache_dir = data_dir / "cache/models/als"
        lfm_cache_dir = data_dir / "cache/models/lightfm"
        ease_cache_dir = data_dir / "cache/models/ease"

        del datasets["train_df_als"]
        del user_disliked_mean_embeddings
        del user_liked_mean_embeddings

        mlflow.log_params({
            "als_iterations": iterations,
            "als_alpha": alpha,
            "als_regularization": regularization,
            "als_n_factors": n_factors,

            "lfm_n_features": lfm_n_features,
            "lfm_n_epochs": lfm_n_epochs,
            "lfm_source_add_n_epochs": lfm_source_add_n_epochs,

            "ease_max_items": ease_max_items,
            "ease_regularization": ease_regularization,
            "ease_max_source_ids": ease_max_source_ids,

            "like_weight_item_like": like_weight_item_like,
        })

        models_like = {
            "als_item_like": ALSModel(
                iterations=iterations,
                alpha=alpha,
                regularization=regularization,
                n_factors=n_factors,
                predict_col_name="predict_als_item_like",
                cache_dir=als_cache_dir,
            ),
            "als_source_like": ALSSource(
                items_meta_df=items_meta_df,
                iterations=iterations,
                alpha=alpha,
                regularization=regularization,
                n_factors=n_factors,
                predict_col_name="predict_als_source_like",
                cache_dir=als_cache_dir,
            ),
            "lfm_item_like": LFMModel(
                n_features=lfm_n_features, 
                n_epochs=lfm_n_epochs, 
                verbose=0,
                predict_col_name="predict_lfm_item_like",
                cache_dir=lfm_cache_dir,
            ),
            "lfm_item_like_warp": LFMModel(
                n_features=lfm_n_features, 
                n_epochs=lfm_n_epochs, 
                loss="warp",
                verbose=0,
                predict_col_name="predict_lfm_item_like_warp",
                cache_dir=lfm_cache_dir,
            ),
            "lfm_source_like": LightFMSource(
                items_meta_df=items_meta_df,
                n_features=lfm_n_features, 
                n_epochs=lfm_n_epochs, 
                verbose=0,
                predict_col_name="predict_lfm_source_like",
                cache_dir=lfm_cache_dir,
            ),
            "lfm_source_like_warp": LightFMSource(
                items_meta_df=items_meta_df,
                n_features=lfm_n_features, 
                n_epochs=lfm_n_epochs, 
                loss="warp",
                verbose=0,
                predict_col_name="predict_lfm_source_like_warp",
                cache_dir=lfm_cache_dir,
            ),
            "lfm_source_add_like": LightFMSourceAdd(
                items_meta_df=items_meta_df,
                n_features=lfm_n_features, 
                n_epochs=lfm_source_add_n_epochs, 
                verbose=0,
                predict_col_name="predict_lfm_source_add_like",
                cache_dir=lfm_cache_dir,
            ),
            "lfm_source_add_like_warp": LightFMSourceAdd(
                items_meta_df=items_meta_df,
                n_features=lfm_n_features, 
                n_epochs=lfm_source_add_n_epochs, 
                loss="warp",
                verbose=0,
                predict_col_name="predict_lfm_source_add_like_warp",
                cache_dir=lfm_cache_dir,
            ),
            "ease_like": EASEModel(
                predict_col_name="predict_ease_like",
                cache_dir=ease_cache_dir,
                max_items=ease_max_items,
                regularization=ease_regularization,
            ),
            "ease_source_like": EASESourceModel(
                items_meta_df=items_meta_df,
                predict_col_name="predict_ease_source_like",
                cache_dir=ease_cache_dir,
                max_items=ease_max_source_ids,
                regularization=ease_regularization,
            ),
        }

        models_like_time_weighted = {
            "als_item_like_time_weighted": ALSModel(
                iterations=iterations,
                alpha=alpha,
                regularization=regularization,
                n_factors=n_factors,
                predict_col_name="predict_als_item_like_time_weighted",
                cache_dir=als_cache_dir,
            ),
            "lfm_item_like_time_weighted": LFMModel(
                n_features=lfm_n_features, 
                n_epochs=lfm_n_epochs, 
                verbose=0,
                predict_col_name="predict_lfm_item_like_time_weighted",
                cache_dir=lfm_cache_dir,
            ),
            "lfm_item_like_warp_time_weighted": LFMModel(
                n_features=lfm_n_features, 
                n_epochs=lfm_n_epochs, 
                loss="warp",
                verbose=0,
                predict_col_name="predict_lfm_item_like_warp_time_weighted",
                cache_dir=lfm_cache_dir,
            ),
        }

        models_like_book_share = {
            "als_item_like_book_share": ALSModel(
                iterations=iterations,
                alpha=alpha,
                regularization=regularization,
                n_factors=n_factors,
                predict_col_name="predict_als_item_like_book_share",
                cache_dir=als_cache_dir,
            ),
            "als_source_like_book_share": ALSSource(
                items_meta_df=items_meta_df,
                iterations=iterations,
                alpha=alpha,
                regularization=regularization,
                n_factors=n_factors,
                predict_col_name="predict_als_source_like_book_share",
                cache_dir=als_cache_dir,
            ),
            "lfm_item_like_book_share": LFMModel(
                n_features=lfm_n_features, 
                n_epochs=lfm_n_epochs, 
                verbose=0,
                predict_col_name="predict_lfm_item_like_book_share",
                cache_dir=lfm_cache_dir,
            ),
            "lfm_item_like_book_share_warp": LFMModel(
                n_features=lfm_n_features, 
                n_epochs=lfm_n_epochs, 
                verbose=0,
                loss="warp",
                predict_col_name="predict_lfm_item_like_book_share_warp",
                cache_dir=lfm_cache_dir,
            ),
            "lfm_source_like_book_share": LightFMSource(
                items_meta_df=items_meta_df,
                n_features=lfm_n_features, 
                n_epochs=lfm_n_epochs, 
                verbose=0,
                predict_col_name="predict_lfm_source_like_book_share",
                cache_dir=lfm_cache_dir,
            ),
            "lfm_source_like_book_share_warp": LightFMSource(
                items_meta_df=items_meta_df,
                n_features=lfm_n_features, 
                n_epochs=lfm_n_epochs, 
                loss="warp",
                verbose=0,
                predict_col_name="predict_lfm_source_like_book_share_warp",
                cache_dir=lfm_cache_dir,
            ),
            "lfm_source_add_like_book_share": LightFMSourceAdd(
                items_meta_df=items_meta_df,
                n_features=lfm_n_features, 
                n_epochs=lfm_source_add_n_epochs, 
                verbose=0,
                predict_col_name="predict_lfm_source_add_like_book_share",
                cache_dir=lfm_cache_dir,
            ),
            "lfm_source_add_like_book_share_warp": LightFMSourceAdd(
                items_meta_df=items_meta_df,
                n_features=lfm_n_features, 
                n_epochs=lfm_source_add_n_epochs, 
                loss="warp",
                verbose=0,
                predict_col_name="predict_lfm_source_add_like_book_share_warp",
                cache_dir=lfm_cache_dir,
            ),
            "ease_like_book_share": EASEModel(
                predict_col_name="predict_ease_like_book_share",
                cache_dir=ease_cache_dir,
                max_items=ease_max_items,
                regularization=ease_regularization,
            ),
            "ease_source_like_book_share": EASESourceModel(
                items_meta_df=items_meta_df,
                predict_col_name="predict_ease_source_like_book_share",
                cache_dir=ease_cache_dir,
                max_items=ease_max_source_ids,
                regularization=ease_regularization,
            ),
        }

        models_like_book_share_time_weighted = {
            "als_item_like_book_share_time_weighted": ALSModel(
                iterations=iterations,
                alpha=alpha,
                regularization=regularization,
                n_factors=n_factors,
                predict_col_name="predict_als_item_like_book_share_time_weighted",
                cache_dir=als_cache_dir,
            ),
            "lfm_item_like_book_share_time_weighted": LFMModel(
                n_features=lfm_n_features, 
                n_epochs=lfm_n_epochs, 
                verbose=0,
                predict_col_name="predict_lfm_item_like_book_share_time_weighted",
                cache_dir=lfm_cache_dir,
            ),
            "lfm_item_like_book_share_warp_time_weighted": LFMModel(
                n_features=lfm_n_features, 
                n_epochs=lfm_n_epochs, 
                verbose=0,
                loss="warp",
                predict_col_name="predict_lfm_item_like_book_share_warp_time_weighted",
                cache_dir=lfm_cache_dir,
            ),
        }

        models_timespent = {
            "als_item_timespent": ALSModel(
                iterations=iterations,
                alpha=alpha,
                regularization=regularization,
                n_factors=n_factors,
                predict_col_name="predict_als_item_timespent",
                cache_dir=als_cache_dir,
            ),
            "lfm_item_timespent": LFMModel(
                n_features=lfm_n_features, 
                n_epochs=lfm_n_epochs, 
                verbose=0,
                predict_col_name="predict_lfm_item_timespent",
                cache_dir=lfm_cache_dir,
            ),
        }

        models_like_all = {
            "lfm_item_like_all": LFMModel(
                n_features=lfm_n_features, 
                n_epochs=lfm_n_epochs, 
                verbose=0,
                predict_col_name="predict_lfm_item_like_all",
                cache_dir=lfm_cache_dir,
            ),
        }

        predicts = {
            "train_df_cb": datasets["train_df_cb"].select("user_id", "item_id"),
            "test_df": datasets["test_df"].select("user_id", "item_id"),
            "test_pairs": test_pairs.select("user_id", "item_id"),
        }

        for model_name, model in models_like.items():
            print(model_name)
            model.fit(train_als_like_item)

            predict_proba_fn = lambda x: model.predict_proba(x) if "ease" not in model_name else model.predict_proba(x, train_als_like_item)

            predicts["train_df_cb"] = (
                predicts["train_df_cb"]
                .join(predict_proba_fn(datasets["train_df_cb"].select("user_id", "item_id")), how="left", on=["user_id", "item_id"])
            )

            predicts["test_df"] = (
                predicts["test_df"]
                .join(predict_proba_fn(datasets["test_df"].select("user_id", "item_id")), how="left", on=["user_id", "item_id"])
            )

            predicts["test_pairs"] = (
                predicts["test_pairs"]
                .join(predict_proba_fn(test_pairs.select("user_id", "item_id")), how="left", on=["user_id", "item_id"])
            )

            del model
        
        matrix_factorization_columns = [model.predict_col_name for _, model in models_like.items()]
        del train_als_like_item
        del models_like
        gc.collect()

        for model_name, model in models_like_book_share.items():
            print(model_name)
            model.fit(train_als_like_book_share_item)

            predict_proba_fn = lambda x: model.predict_proba(x) if "ease" not in model_name else model.predict_proba(x, train_als_like_book_share_item)

            predicts["train_df_cb"] = (
                predicts["train_df_cb"]
                .join(predict_proba_fn(datasets["train_df_cb"].select("user_id", "item_id")), how="left", on=["user_id", "item_id"])
            )

            predicts["test_df"] = (
                predicts["test_df"]
                .join(predict_proba_fn(datasets["test_df"].select("user_id", "item_id")), how="left", on=["user_id", "item_id"])
            )

            predicts["test_pairs"] = (
                predicts["test_pairs"]
                .join(predict_proba_fn(test_pairs.select("user_id", "item_id")), how="left", on=["user_id", "item_id"])
            )

            del model

        matrix_factorization_columns.extend([model.predict_col_name for _, model in models_like_book_share.items()])
        
        del train_als_like_book_share_item
        del models_like_book_share
        gc.collect()

        for model_name, model in models_like_all.items():
            print(model_name)
            model.fit(train_als_like_item_all)

            predict_proba_fn = lambda x: model.predict_proba(x)

            predicts["train_df_cb"] = (
                predicts["train_df_cb"]
                .join(predict_proba_fn(datasets["train_df_cb"].select("user_id", "item_id")), how="left", on=["user_id", "item_id"])
            )

            predicts["test_df"] = (
                predicts["test_df"]
                .join(predict_proba_fn(datasets["test_df"].select("user_id", "item_id")), how="left", on=["user_id", "item_id"])
            )

            predicts["test_pairs"] = (
                predicts["test_pairs"]
                .join(predict_proba_fn(test_pairs.select("user_id", "item_id")), how="left", on=["user_id", "item_id"])
            )

            del model

        matrix_factorization_columns.extend([model.predict_col_name for _, model in models_like_all.items()])
        
        del train_als_like_item_all
        del models_like_all
        gc.collect()

        for model_name, model in models_timespent.items():
            print(model_name)
            model.fit(train_als_timespent)

            predicts["train_df_cb"] = (
                predicts["train_df_cb"]
                .join(model.predict_proba(datasets["train_df_cb"].select("user_id", "item_id")), how="left", on=["user_id", "item_id"])
            )

            predicts["test_df"] = (
                predicts["test_df"]
                .join(model.predict_proba(datasets["test_df"].select("user_id", "item_id")), how="left", on=["user_id", "item_id"])
            )

            predicts["test_pairs"] = (
                predicts["test_pairs"]
                .join(model.predict_proba(test_pairs.select("user_id", "item_id")), how="left", on=["user_id", "item_id"])
            )

            del model

        matrix_factorization_columns.extend([model.predict_col_name for _, model in models_timespent.items()])

        del train_als_timespent
        del models_timespent
        gc.collect()

        for model_name, model in models_like_book_share_time_weighted.items():
            print(model_name)
            model.fit(train_als_like_book_share_item_time_weighted)

            predicts["train_df_cb"] = (
                predicts["train_df_cb"]
                .join(model.predict_proba(datasets["train_df_cb"].select("user_id", "item_id")), how="left", on=["user_id", "item_id"])
            )

            predicts["test_df"] = (
                predicts["test_df"]
                .join(model.predict_proba(datasets["test_df"].select("user_id", "item_id")), how="left", on=["user_id", "item_id"])
            )

            predicts["test_pairs"] = (
                predicts["test_pairs"]
                .join(model.predict_proba(test_pairs.select("user_id", "item_id")), how="left", on=["user_id", "item_id"])
            )

            del model

        matrix_factorization_columns.extend([model.predict_col_name for _, model in models_like_book_share_time_weighted.items()])
        
        del train_als_like_book_share_item_time_weighted
        del models_like_book_share_time_weighted
        gc.collect()

        for model_name, model in models_like_time_weighted.items():
            print(model_name)
            model.fit(train_als_like_item_time_weighted)

            predicts["train_df_cb"] = (
                predicts["train_df_cb"]
                .join(model.predict_proba(datasets["train_df_cb"].select("user_id", "item_id")), how="left", on=["user_id", "item_id"])
            )

            predicts["test_df"] = (
                predicts["test_df"]
                .join(model.predict_proba(datasets["test_df"].select("user_id", "item_id")), how="left", on=["user_id", "item_id"])
            )

            predicts["test_pairs"] = (
                predicts["test_pairs"]
                .join(model.predict_proba(test_pairs.select("user_id", "item_id")), how="left", on=["user_id", "item_id"])
            )

            del model
        
        matrix_factorization_columns.extend([model.predict_col_name for _, model in models_like_time_weighted.items()])

        del train_als_like_item_time_weighted
        del models_like_time_weighted
        gc.collect()

        train_df_cb_final = join_features(
            datasets["train_df_cb"],
            predicts["train_df_cb"],
            item_stats,
            source_stats,
            items_meta_df,
            train_df_cb_sim_features,
            users_meta_df=users_meta_df,
            user_stats=user_stats,
            user2source_stats=user2source_stats,
        ).with_columns(
            (pl.col("like").cast(int) - pl.col("dislike").cast(int)).alias("target")
        )

        test_df_final = join_features(
            datasets["test_df"],
            predicts["test_df"],
            item_stats,
            source_stats,
            items_meta_df,
            test_df_sim_features,
            users_meta_df=users_meta_df,
            user_stats=user_stats,
            user2source_stats=user2source_stats,
        ).with_columns(
            (pl.col("like").cast(int) - pl.col("dislike").cast(int)).alias("target")
        )

        # test_df_final, val_df_final = train_test_split_by_user_id(test_df_final, 0.2)
        val_df_final = None

        test_pairs_final = join_features(
            test_pairs,
            predicts["test_pairs"],
            item_stats,
            source_stats,
            items_meta_df,
            test_pairs_sim_features,
            users_meta_df=users_meta_df,
            user_stats=user_stats,
            user2source_stats=user2source_stats,
        )

        del item_stats
        del source_stats
        # del items_meta_df
        del users_meta_df
        del user_stats
        del train_df_cb_sim_features
        del test_df_sim_features
        del test_pairs_sim_features
        del predicts

        gc.collect()

        feature_columns_raw = [c for c in test_pairs_final.columns if c not in ("user_id", "item_id")]
        
        train_df_cb_final, feature_columns = add_poly_features(train_df_cb_final, feature_columns_raw)
        if save_datasets:
            train_df_cb_final.write_parquet("./data/catboost_dataset_train.parquet")
        train_df_cb_final = train_df_cb_final.to_pandas()

        cb_iterations = 1000
        cb_depth = 6

        cb_loss_function = "YetiRank"
        mlflow.log_params({
            "cb_iterations": cb_iterations,
            "cb_depth": cb_depth,
            "cb_loss_function": cb_loss_function,
        })

        item_embeddings_df = items_meta_df[["item_id", "embeddings"]].to_pandas()

        train_df_cb_final = pd.merge(
            train_df_cb_final,
            item_embeddings_df,
            on="item_id", how="left"
        )

        feature_columns = list(feature_columns)
        feature_columns.append("embeddings")

        train_pool = get_cb_pool(train_df_cb_final, feature_columns)

        val_pool = None
        if val_df_final is not None:
            val_df_final, _ = add_poly_features(val_df_final, feature_columns_raw)
            if save_datasets:
                val_df_final.write_parquet("./data/catboost_dataset_val.parquet")
            val_df_final = val_df_final.to_pandas()

            val_df_final = pd.merge(
                val_df_final,
                item_embeddings_df,
                on="item_id", how="left"
            )
            val_pool = get_cb_pool(val_df_final, feature_columns)

        logger.info("training catboost model")
        cb_model = CatBoostRanker(
            iterations=cb_iterations, 
            depth=cb_depth, 
            random_seed=32, 
            verbose=50, 
            # colsample_bylevel=0.8,
            # subsample=0.8,
            loss_function=cb_loss_function,
            eval_metric="QueryAUC:type=Ranking",
            early_stopping_rounds=50,
        )
        cb_model.fit(train_pool, eval_set=val_pool)

        del train_df_cb_final
        del train_pool

        test_df_final, _ = add_poly_features(test_df_final, feature_columns_raw)
        if save_datasets:
            test_df_final.write_parquet("./data/catboost_dataset_test.parquet")
        test_df_final = test_df_final.to_pandas()

        test_df_final = pd.merge(
            test_df_final,
            item_embeddings_df,
            on="item_id", how="left"
        )
        test_pool = get_cb_pool(test_df_final, feature_columns)
        
        test_predict = cb_model.predict(test_pool)
        test_df_final_prediction = (
            pl.from_pandas(test_df_final[["user_id", "target", "item_id", *matrix_factorization_columns]])
            .with_columns(
                pl.Series(test_predict).alias("prediction")
            )
        )

        del test_df_final
        del test_pool

        metric_value = calc_user_auc(
            df=test_df_final_prediction,
            predict_col="prediction", 
            target_col="target"
        )["rocauc"].mean()

        for predict_col in matrix_factorization_columns + ["prediction", ]:
            metric_value = calc_user_auc(
                df=test_df_final_prediction,
                predict_col=predict_col, 
                target_col="target"
            )["rocauc"].mean()

            print(f"{predict_col}: {metric_value:.5f}")
            mlflow.log_metric(f"{predict_col}_rocauc", metric_value)

        test_pairs_final, _ = add_poly_features(test_pairs_final, feature_columns_raw)
        test_pairs_final = test_pairs_final.to_pandas()
        test_pairs_final = pd.merge(
            test_pairs_final,
            item_embeddings_df,
            on="item_id", how="left"
        )
        test_pairs_pool = get_cb_pool(test_pairs_final, feature_columns, False)

        submission_predict = cb_model.predict(test_pairs_pool)
        test_pairs_final["predict"] = submission_predict
        submission_path = data_dir / f'submissions/{int(datetime.now().timestamp())}_submission.csv'
        submission_path = submission_path.as_posix()
        test_pairs_final[["user_id", "item_id", "predict"]].to_csv(submission_path, index=False)

        mlflow.log_param("submission_path", submission_path)

        mlflow.log_params({
            "feature_columns_raw": feature_columns_raw,
            "feature_columns": feature_columns,
            "len_feature_columns": len(feature_columns),
        })


if __name__ == "__main__":
    cli()
