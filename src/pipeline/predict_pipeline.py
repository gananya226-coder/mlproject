import os
import pickle

import pandas as pd

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")


def load_object(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


class PredictPipeline:
    """Loads the trained model + preprocessor and predicts math_score."""

    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        model = load_object(MODEL_PATH)
        preprocessor = load_object(PREPROCESSOR_PATH)

        data_scaled = preprocessor.transform(features)
        preds = model.predict(data_scaled)
        return preds


class CustomData:
    """Maps the web form inputs to the columns used during training (stud.csv)."""

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        data = {
            "gender": [self.gender],
            "race_ethnicity": [self.race_ethnicity],
            "parental_level_of_education": [self.parental_level_of_education],
            "lunch": [self.lunch],
            "test_preparation_course": [self.test_preparation_course],
            "reading_score": [self.reading_score],
            "writing_score": [self.writing_score],
        }
        return pd.DataFrame(data)