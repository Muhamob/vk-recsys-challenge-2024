import polars as pl
from implicit.als import AlternatingLeastSquares
from rectools.models.lightfm import LightFM, LightFMWrapperModel
from rectools import Columns
from rectools.dataset import Dataset

from src.logger import logger


class LFMModel:
    num_threads = 8

    def __init__(
        self,
        n_features: int = 10,
        n_epochs: int = 10,
        loss: str = "bpr",
        predict_col_name: str = "predict",
        cold_predict: float = -1.0,
        verbose: int = 0,
    ):
        self.predict_col_name = predict_col_name
        self.cold_predict = cold_predict
        self.loss = loss
        self.n_features = n_features
        self.n_epochs = n_epochs

        self.model = LightFMWrapperModel(
            LightFM(no_components=n_features, random_state=42, loss=loss,),
            num_threads=self.num_threads, verbose=verbose, epochs=self.n_epochs
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

        del self.model

        return self
    
    def predict_proba(self, pairs_df: pl.DataFrame) -> pl.DataFrame:
        assert self.is_fitted
        assert self.user2idx is not None
        assert self.item2idx is not None
        assert self.user_embeddings is not None
        assert self.item_embeddings is not None
        assert "user_id" in pairs_df.columns
        assert "item_id" in pairs_df.columns

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

        hot_pairs = (
            hot_pairs
            .with_columns(
                pl.lit(user2item_dist).cast(pl.Float32).alias(self.predict_col_name)
            )
        )

        cold_pairs = (
            pairs_df
            .join(hot_pairs.select("user_id", "item_id"), how="anti", on=["user_id", "item_id"])
            .with_columns(
                pl.lit(self.cold_predict).cast(pl.Float32).alias(self.predict_col_name)
            )
        )

        logger.info(f"ALS: Percent of cold pairs: {float(cold_pairs.shape[0]) / pairs_df.shape[0]}")

        return pl.concat([hot_pairs, cold_pairs])
