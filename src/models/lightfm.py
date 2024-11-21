from pathlib import Path

import polars as pl
from rectools.models.lightfm import LightFM, LightFMWrapperModel
from rectools.dataset import Dataset

from src.logger import logger
from src.models.base_matrix_factorization import BaseMatrixFactorization


class LFMModel(BaseMatrixFactorization):
    num_threads = 8

    def __init__(
        self,
        n_features: int = 10,
        n_epochs: int = 10,
        loss: str = "bpr",
        predict_col_name: str = "predict",
        cold_predict: float = -1.0,
        verbose: int = 0,
        random_state: int = 42,
        cache_dir: Path | None = None,
    ):
        self.predict_col_name = predict_col_name
        self.cold_predict = cold_predict
        self.loss = loss
        self.n_features = n_features
        self.n_epochs = n_epochs
        self.random_state = random_state

        self.model = LightFMWrapperModel(
            LightFM(no_components=n_features, random_state=random_state, loss=loss,),
            num_threads=self.num_threads, verbose=verbose, epochs=self.n_epochs
        )

        super().__init__(
            cache_dir=cache_dir,
            # params
            predict_col_name=predict_col_name,
            cold_predict=cold_predict,
            loss=loss,
            n_features=n_features,
            n_epochs=n_epochs,
            random_state=random_state,
            return_user_bias=True,
            return_item_bias=True,
            add_user_and_item_bias=True,
        )

        self.user2idx = None
        self.nmid2idx = None
        self.user_embeddings = None
        self.item_embeddings = None

        self.is_fitted = self.model.is_fitted

    def fit(
        self, 
        train_df: pl.DataFrame,
        items_meta_df_flatten: pl.DataFrame | None = None,
        users_meta_df_flatten: pl.DataFrame | None = None,
    ):
        load_cache_result = self._try_load_cache(train_df, items_meta_df_flatten, users_meta_df_flatten)
        if load_cache_result:
            return self
        
        assert self.model is not None
        
        additional_params = {}

        if items_meta_df_flatten is not None:
            additional_params.update({
                "item_features_df": items_meta_df_flatten.to_pandas(),
                "cat_item_features": items_meta_df_flatten["feature"].unique().to_list(),
            })

        if users_meta_df_flatten is not None:
            additional_params.update({
                "user_features_df": users_meta_df_flatten.to_pandas(),
                "cat_user_features": users_meta_df_flatten["feature"].unique().to_list(),
            })
        
        dataset = Dataset.construct(
            interactions_df=train_df.to_pandas(),
            **additional_params
        )

        self.user2idx = dataset.user_id_map
        self.item2idx = dataset.item_id_map

        self.model.fit(dataset)
        
        self.user_embeddings, self.item_embeddings = self.model.get_vectors(dataset=dataset, add_biases=False)
        self.is_fitted = self.model.is_fitted

        self._save_cache(train_df, items_meta_df_flatten, users_meta_df_flatten)

        return self
    
    def predict_proba(self, pairs_df: pl.DataFrame) -> pl.DataFrame:
        assert self.is_fitted
        assert self.user2idx is not None
        assert self.item2idx is not None
        assert self.user_embeddings is not None
        assert self.item_embeddings is not None
        assert "user_id" in pairs_df.columns
        assert "item_id" in pairs_df.columns
        assert self.model is not None
        assert self.model.is_fitted

        catalog = self.item2idx.external_ids

        hot_pairs = (
            pairs_df
            .filter(pl.col("user_id").is_in(set(self.user2idx.external_ids)))
            .filter(pl.col("item_id").is_in(set(catalog)))
        )

        assert catalog.shape[0] == self.item_embeddings.shape[0]

        user2item_dist = (
            self.user_embeddings[self.user2idx.convert_to_internal(hot_pairs["user_id"].to_numpy())]
            * self.item_embeddings[self.item2idx.convert_to_internal(hot_pairs["item_id"].to_numpy())]
        ).sum(axis=1)

        user_biases = self.model.model.user_biases[self.user2idx.convert_to_internal(hot_pairs["user_id"].to_numpy())]
        item_biases = self.model.model.item_biases[self.item2idx.convert_to_internal(hot_pairs["item_id"].to_numpy())]

        hot_pairs = (
            hot_pairs
            .with_columns(
                pl.lit(user2item_dist).cast(pl.Float32).alias(self.predict_col_name),
                pl.lit(user2item_dist + user_biases + item_biases).cast(pl.Float32).alias(self.predict_col_name + "_with_user_and_item_bias"),
                pl.lit(user_biases).cast(pl.Float32).alias(self.predict_col_name + "_only_user_bias"),
                pl.lit(item_biases).cast(pl.Float32).alias(self.predict_col_name + "_only_item_bias"),
            )
        )

        cold_pairs = (
            pairs_df
            .join(hot_pairs.select("user_id", "item_id"), how="anti", on=["user_id", "item_id"])
            .with_columns(
                pl.lit(self.cold_predict).cast(pl.Float32).alias(self.predict_col_name),
                pl.lit(self.cold_predict).cast(pl.Float32).alias(self.predict_col_name + "_with_user_and_item_bias"),
                pl.lit(self.cold_predict).cast(pl.Float32).alias(self.predict_col_name + "_only_user_bias"),
                pl.lit(self.cold_predict).cast(pl.Float32).alias(self.predict_col_name + "_only_item_bias"),
            )
        )

        logger.info(f"ALS: Percent of cold pairs: {float(cold_pairs.shape[0]) / pairs_df.shape[0]}")

        return pl.concat([hot_pairs, cold_pairs])
