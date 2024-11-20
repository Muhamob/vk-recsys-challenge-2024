from datetime import datetime
import gc
from itertools import product
import os
from pathlib import Path
from typing import Sequence
import optuna
import threadpoolctl
import logging

import polars as pl
import numpy as np
from tqdm import tqdm
from catboost import CatBoostClassifier, CatBoostRanker
import click
import mlflow

from src.data.item_stats import get_item_stats
from src.data.user_stats import get_user_stats
from src.metrics import calc_user_auc
from src.models.als import ALSModel
from src.models.als_source import ALSSource
from src.models.lightfm import LFMModel
from src.models.lightfm_source import LightFMSource
from src.models.w2v_model import W2VModel
from src.data.preprocessing import load_data, prepare_train_for_als_item_like, prepare_train_for_als_item_like_book_share, prepare_train_for_als_timespent
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
):
    df = (
        target
        .join(predict, how="left", on=["user_id", "item_id"])
        .join(item_stats, how="left", on=["item_id"], suffix="_item")
        .join(items_meta_df.select("item_id", "source_id"), how="left", on="item_id")
        .join(source_stats, how="left", on="source_id", suffix="_source")
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
                (pl.col("age") - pl.col(col)).alias(f"{col}_diff_abs")
                for col in age_features
            ])
        )
        
        gender_features = [col for col in df.columns if "gender" in col]
        gender_features = [col for col in age_features if col != "gender"]
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

    # extra features
    df = (
        df
        .with_columns(
            (pl.col("user_like_perc") * pl.col("mean_like")).alias("user2item_like_perc")
        )
    )

    return (
        df
        .sort("user_id")
    )


@click.group()
@click.option("--level", default="info")
def cli(level):
    logger.setLevel({
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "error": logging.ERROR,
    }[level])


def get_training_datasets(data_dir: Path):
    data_dir = Path(data_dir)

    datasets = load_data(data_dir / "processed")
    test_pairs = pl.read_csv(data_dir / "raw/test_pairs.csv.csv").with_columns(
        pl.col("item_id").cast(pl.UInt32),
        pl.col("user_id").cast(pl.UInt32)
    )
    items_meta_df = pl.read_parquet(data_dir / "raw/items_meta.parquet")
    users_meta_df = pl.read_parquet(data_dir / "raw/users_meta.parquet")

    train_als_like_item = prepare_train_for_als_item_like(datasets["train_df_als"])
    train_als_like_book_share_item = prepare_train_for_als_item_like_book_share(datasets["train_df_als"])
    train_als_timespent = prepare_train_for_als_timespent(datasets["train_df_als"], items_meta_df=items_meta_df)

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

    # w2v
    w2v_n_features = 128
    w2v_n_epochs = 10

    als_cache_dir = data_dir / "cache/models/als"
    lfm_cache_dir = data_dir / "cache/models/lightfm"
    w2v_cache_dir = data_dir / "cache/models/w2v"

    del datasets["train_df_als"]
    del user_disliked_mean_embeddings
    del user_liked_mean_embeddings

    models_w2v = {
        "like": W2VModel(
            n_features=w2v_n_features, 
            n_epochs=w2v_n_epochs, 
            predict_col_name="w2v_predict_like", 
            cache_dir=w2v_cache_dir
        ),
        "timespent": W2VModel(
            n_features=w2v_n_features, 
            n_epochs=w2v_n_epochs, 
            predict_col_name="w2v_predict_timespent", 
            cache_dir=w2v_cache_dir
        ),
    }

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
        )
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

    predicts = {
        "train_df_cb": datasets["train_df_cb"].select("user_id", "item_id"),
        "test_df": datasets["test_df"].select("user_id", "item_id"),
        "test_pairs": test_pairs.select("user_id", "item_id"),
    }

    for model_name, model in models_like.items():
        print(model_name)
        model.fit(train_als_like_item)

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
    
    matrix_factorization_columns = [model.predict_col_name for _, model in models_like.items()]

    # model_w2v = models_w2v["like"]
    # model_w2v.fit(train_als_like_item)
    # predicts["train_df_cb"] = (
    #     predicts["train_df_cb"]
    #     .join(model_w2v.predict_proba(
    #         datasets["train_df_cb"].select("user_id", "item_id"), 
    #         train_als_like_item
    #     ), how="left", on=["user_id", "item_id"])
    # )
    # predicts["test_df"] = (
    #     predicts["test_df"]
    #     .join(model_w2v.predict_proba(
    #         datasets["test_df"].select("user_id", "item_id"), 
    #         train_als_like_item
    #     ), how="left", on=["user_id", "item_id"])
    # )
    # predicts["test_pairs"] = (
    #     predicts["test_pairs"]
    #     .join(model_w2v.predict_proba(
    #         test_pairs.select("user_id", "item_id"), 
    #         train_als_like_item
    #     ), how="left", on=["user_id", "item_id"])
    # )
    # matrix_factorization_columns.append(model_w2v.predict_col_name)

    del models_w2v["like"]
    del train_als_like_item
    del models_like
    gc.collect()

    for model_name, model in models_like_book_share.items():
        print(model_name)
        model.fit(train_als_like_book_share_item)

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

    matrix_factorization_columns.extend([model.predict_col_name for _, model in models_like_book_share.items()])
    
    del train_als_like_book_share_item
    del models_like_book_share
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

    # model_w2v = models_w2v["timespent"]
    # model_w2v.fit(train_als_timespent)
    # predicts["train_df_cb"] = (
    #     predicts["train_df_cb"]
    #     .join(model_w2v.predict_proba(
    #         datasets["train_df_cb"].select("user_id", "item_id"), 
    #         train_als_timespent
    #     ), how="left", on=["user_id", "item_id"])
    # )
    # predicts["test_df"] = (
    #     predicts["test_df"]
    #     .join(model_w2v.predict_proba(
    #         datasets["test_df"].select("user_id", "item_id"), 
    #         train_als_timespent
    #     ), how="left", on=["user_id", "item_id"])
    # )
    # predicts["test_pairs"] = (
    #     predicts["test_pairs"]
    #     .join(model_w2v.predict_proba(
    #         test_pairs.select("user_id", "item_id"), 
    #         train_als_timespent
    #     ), how="left", on=["user_id", "item_id"])
    # )
    # matrix_factorization_columns.append(model_w2v.predict_col_name)

    del models_w2v["timespent"]
    del train_als_timespent
    del models_timespent
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
    ).with_columns(
        (pl.col("like").cast(int) - pl.col("dislike").cast(int)).alias("target")
    )

    test_pairs_final = join_features(
        test_pairs,
        predicts["test_pairs"],
        item_stats,
        source_stats,
        items_meta_df,
        test_pairs_sim_features,
        users_meta_df=users_meta_df,
        user_stats=user_stats,
    )

    del item_stats
    del source_stats
    del items_meta_df
    del users_meta_df
    del user_stats
    del train_df_cb_sim_features
    del test_df_sim_features
    del test_pairs_sim_features
    del predicts

    gc.collect()

    feature_columns = [c for c in test_pairs_final.columns if c not in ("user_id", "item_id")]

    return {
        "feature_columns": feature_columns,
        "train_df_cb_final": train_df_cb_final.to_pandas(),
        "test_df_final": test_df_final.to_pandas(),
        "test_pairs_final": test_pairs_final.to_pandas(),
        "matrix_factorization_columns": matrix_factorization_columns,
        "log_params": {
            "als_iterations": iterations,
            "als_alpha": alpha,
            "als_regularization": regularization,
            "als_n_factors": n_factors,

            "lfm_n_features": lfm_n_features,
            "lfm_n_epochs": lfm_n_epochs,

            "w2v_n_features": w2v_n_features,
            "w2v_n_epochs": w2v_n_epochs,

            "feature_columns": feature_columns,
            "len_feature_columns": len(feature_columns),
        }
    }


def optimize_(trial, data):
    with mlflow.start_run():
        mlflow.log_params(data["log_params"])
        cb_iterations = trial.suggest_int('iterations', 400, 1500)
        cb_depth = trial.suggest_int('depth', 5, 7)
        cb_grow_policy = trial.suggest_categorical("grow_policy", [
            "SymmetricTree", "Depthwise", "Lossguide"
        ])
        cb_border_count = trial.suggest_categorical("border_count", [32, 64, 96])
        cb_l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 0.001, 0.5)
        cb_loss_function = "YetiRank"

        mlflow.log_params({
            "cb_iterations": cb_iterations,
            "cb_depth": cb_depth,
            "cb_loss_function": cb_loss_function,
            "cb_grow_policy": cb_grow_policy,
            "cb_l2_leaf_reg": cb_l2_leaf_reg,
            "cb_border_count": cb_border_count,
        })

        cb_model = CatBoostRanker(
            iterations=cb_iterations, 
            depth=cb_depth, 
            random_seed=32, 
            verbose=0, 
            loss_function=cb_loss_function,
            grow_policy=cb_grow_policy,
            l2_leaf_reg=cb_l2_leaf_reg,
            border_count=cb_border_count,
        )
        train_df_cb_final = data["train_df_cb_final"]
        cb_model.fit(
            train_df_cb_final[data["feature_columns"]], 
            train_df_cb_final["target"], 
            group_id=train_df_cb_final["user_id"]
        )

        test_df_final = data["test_df_final"]
        test_predict = cb_model.predict(test_df_final[data["feature_columns"]])
        test_df_final_prediction = (
            pl.from_pandas(test_df_final[["user_id", "target", "item_id", *data["matrix_factorization_columns"]]])
            .with_columns(
                pl.Series(test_predict).alias("prediction")
            )
        )

        for predict_col in data["matrix_factorization_columns"] + ["prediction", ]:
            metric_value = calc_user_auc(
                df=test_df_final_prediction,
                predict_col=predict_col, 
                target_col="target"
            )["rocauc"].mean()

            print(f"{predict_col}: {metric_value:.5f}")
            mlflow.log_metric(f"{predict_col}_rocauc", metric_value)

        metric_value = calc_user_auc(
            df=test_df_final_prediction,
            predict_col="prediction", 
            target_col="target"
        )["rocauc"].mean()
    
    return metric_value


@cli.command()
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False), default="./data/")
def optimize(data_dir: Path):
    # PYTHONPATH=. python scripts/train.py --level debug optimize --data-dir ./data
    mlflow.set_experiment("vk")
    mlflow.set_tracking_uri("http://127.0.0.1:8080") 

    data = get_training_datasets(data_dir)
    
    study_cb = optuna.create_study(direction="maximize")
    study_cb.optimize(lambda x: optimize_(x, data), n_trials=100)
    study_cb.best_params


if __name__ == "__main__":
    cli()
