import pandas as pd
import polars as pl
from catboost import CatBoostClassifier, CatBoostRanker
from sklearn.model_selection import train_test_split

from src.logger import logger


class CatBoostRankerModel:
    def __init__(
        self, 
        model_params: dict, 
        feature_columns: list, 
        target_col: str = "target", 
        group_col: str = "user_id"
    ):
        self.model_params = model_params
        self.model = CatBoostRanker(**self.model_params)

        self.feature_columns = feature_columns
        self.target_col = target_col
        self.group_col = group_col

    def fit(self, df: pd.DataFrame):
        logger.debug(f"Training catboost ranker model: {len(self.feature_columns)} features, {self.target_col} target")
        self.model.fit(
            df[self.feature_columns], 
            df[self.target_col], 
            group_id=df[self.group_col]
        )
        return self

    def predict(self, df: pd.DataFrame):
        return self.model.predict(df[self.feature_columns])


class CatBoostBinaryClassifierModel:
    def __init__(
        self, 
        model_params: dict,
        feature_columns: list, 
        target_col: str = "target", 
    ):
        self.model_params = model_params
        self.model = CatBoostClassifier(**self.model_params)
        self.feature_columns = feature_columns
        self.target_col = target_col

    def fit(self, df: pd.DataFrame):
        logger.debug(f"Training catboost classifier model: {len(self.feature_columns)} features, {self.target_col} target")
        self.model.fit(df[self.feature_columns], df[self.target_col])
        return self

    def predict(self, df: pd.DataFrame):
        return self.model.predict_proba(df[self.feature_columns])[:, 1]
    

class CBBlendingRanker:
    def __init__(
        self, 
        models: list, 
        final_model_params: dict,
        test_size: float = 0.1, 
        random_state: int = 422
    ):
        self.models = models
        self.columns = [f"col_{i}" for i in range(len(models))]
        self.test_size = test_size
        self.random_state = random_state
        self.final_model_params = final_model_params
        self.final_model = CatBoostRanker(**self.final_model_params)

    def fit(self, df: pd.DataFrame):
        user_ids = df["user_id"].unique()
        train_uids, test_uids = train_test_split(user_ids, random_state=self.random_state, test_size=self.test_size)

        train_df = pd.merge(pd.DataFrame({"user_id": train_uids}), df, how="inner", on="user_id").sort_values(["user_id", "item_id"])
        test_df = pd.merge(pd.DataFrame({"user_id": test_uids}), df, how="inner", on="user_id").sort_values(["user_id", "item_id"])

        predictions = test_df[["user_id", "item_id", "target"]]
        for model, col in zip(self.models, self.columns):
            model.fit(train_df)
            predictions[col] = model.predict(test_df)

        self.final_model.fit(predictions[self.columns], predictions["target"], predictions["user_id"])
        
        return self

    def predict(self, df: pd.DataFrame, return_features: bool = False):
        prediction = df[["user_id", "item_id"]]
        for col, model in zip(self.columns, self.models):
            prediction[col] = model.predict(df)

        predict = self.final_model.predict(prediction[self.columns])

        if return_features:
            return predict, prediction 

        return predict