from datetime import datetime
from pathlib import Path

import polars as pl
from tqdm import tqdm
from catboost import CatBoostRanker
import click

from src.data.item_stats import get_item_stats
from src.metrics import calc_user_auc
from src.models.als import ALSModel
from src.models.als_item import ALSSource
from src.data.preprocessing import load_data, prepare_train_for_als_item_like, prepare_train_for_als_item_like_book_share


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
):
    df = (
        target
        .join(predict, how="left", on=["user_id", "item_id"])
        .join(item_stats, how="left", on=["item_id"], suffix="_item")
        .join(items_meta_df.select("item_id", "source_id"), how="left", on="item_id")
        .join(source_stats, how="left", on="source_id", suffix="_source")
    )

    if sim_features is not None:
        df = df.join(sim_features, how="left", on=["user_id", "item_id"])

    return (
        df
        .sort("user_id")
    )


@click.command()
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False), default="./data/")
def main(data_dir: Path):
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

    item_stats = get_item_stats(datasets["train_df_als"], items_meta_df, column="item_id")
    source_stats = get_item_stats(datasets["train_df_als"], items_meta_df, column="source_id")

    users_meta_df_flatten = (
        users_meta_df
        .with_columns(pl.col("age").cut(list(range(10, 100, 5))).to_physical().alias("age_group"))
        .melt(id_vars=["user_id"], value_vars=["gender", "age_group"], variable_name="feature", value_name="value")
        .rename({"user_id": "id"})
    )

    items_meta_df_flatten = (
        items_meta_df
        .with_columns(
            pl.col("duration").cut([25, 60]).to_physical()
        )
        .melt(id_vars=["item_id"], value_vars=["duration"], variable_name="feature", value_name="value")
        .rename({"item_id": "id"})
    )

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
    n_factors = 96

    models_like = {
        "als_item_like": ALSModel(
            iterations=iterations,
            alpha=alpha,
            regularization=regularization,
            n_factors=n_factors,
            predict_col_name="predict_als_item_like"
        ),
        "als_source_like": ALSSource(
            items_meta_df=items_meta_df,
            iterations=iterations,
            alpha=alpha,
            regularization=regularization,
            n_factors=n_factors,
            predict_col_name="predict_als_source_like"
        ),
    }

    models_like_book_share = {
        "als_item_like_book_share": ALSModel(
            iterations=iterations,
            alpha=alpha,
            regularization=regularization,
            n_factors=n_factors,
            predict_col_name="predict_als_item_like_book_share"
        ),
        "als_source_like_book_share": ALSSource(
            items_meta_df=items_meta_df,
            iterations=iterations,
            alpha=alpha,
            regularization=regularization,
            n_factors=n_factors,
            predict_col_name="predict_als_source_like_book_share"
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

    train_df_cb_final = join_features(
        datasets["train_df_cb"],
        predicts["train_df_cb"],
        item_stats,
        source_stats,
        items_meta_df,
        train_df_cb_sim_features,
    ).with_columns(
        (pl.col("like").cast(int) - pl.col("dislike").cast(int)).alias("target")
    ).to_pandas()

    test_df_final = join_features(
        datasets["test_df"],
        predicts["test_df"],
        item_stats,
        source_stats,
        items_meta_df,
        test_df_sim_features
    ).with_columns(
        (pl.col("like").cast(int) - pl.col("dislike").cast(int)).alias("target")
    ).to_pandas()

    test_pairs_final = join_features(
        test_pairs,
        predicts["test_pairs"],
        item_stats,
        source_stats,
        items_meta_df,
        test_pairs_sim_features
    ).to_pandas()

    feature_columns = [c for c in test_pairs_final.columns if c not in ("user_id", "item_id")]
    print(feature_columns)

    cb_model = CatBoostRanker(
        iterations=1000, 
        depth=6, 
        random_seed=32, 
        verbose=0, 
        loss_function="YetiRank"
    )
    cb_model.fit(train_df_cb_final[feature_columns], train_df_cb_final["target"], group_id=train_df_cb_final["user_id"])
    test_predict = cb_model.predict(test_df_final[feature_columns])

    als_columns = ['predict_als_item_like', 'predict_als_source_like', 'predict_als_item_like_book_share', 'predict_als_source_like_book_share']

    test_df_final_prediction = (
        pl.from_pandas(test_df_final[["user_id", "target", "item_id", *als_columns]])
        .with_columns(
            pl.Series(test_predict).alias("prediction")
        )
    )

    metric_value = calc_user_auc(
        df=test_df_final_prediction,
        predict_col="prediction", 
        target_col="target"
    )["rocauc"].mean()

    for predict_col in als_columns + ["prediction", ]:
        metric_value = calc_user_auc(
            df=test_df_final_prediction,
            predict_col=predict_col, 
            target_col="target"
        )["rocauc"].mean()

        print(f"{predict_col}: {metric_value:.5f}")

    submission_predict = cb_model.predict(test_pairs_final[feature_columns])
    test_pairs_final["predict"] = submission_predict
    test_pairs_final[["user_id", "item_id", "predict"]].to_csv(data_dir / f'submissions/{int(datetime.now().timestamp())}_submission.csv', index=False)


if __name__ == "__main__":
    main()
