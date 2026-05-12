import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .dataset import Dataset

class DatasetPreprocessor:
    def _preprocess_wine(self):
        self.dataset.df = self.dataset.df.rename(
            columns={
                "fixed acidity": "fixed_acidity",
                "volatile acidity": "volatile_acidity",
                "citric acid": "citric_acid",
                "residual sugar": "residual_sugar",
                "free sulfur dioxide": "free_sulfur_dioxide",
                "total sulfur dioxide": "total_sulfur_dioxide",
            }
        )

    def _preprocess_breast_cancer(self):
        self.dataset.df.rename(
            columns={
                "concave points_mean": "concave_points_mean",
                "concave points_worst": "concave_points_worst",
                "concave points_se": "concave_points_se",
            },
            inplace=True,
        )
        self.dataset.df.drop(["id"], axis=1, inplace=True)
        self.dataset.df = self.dataset.df[
            [
                "radius_mean",
                "texture_mean",
                "perimeter_mean",
                "area_mean",
                "smoothness_mean",
                "compactness_mean",
                "concavity_mean",
                "concave_points_mean",
                "symmetry_mean",
                "fractal_dimension_mean",
                "radius_se",
                "texture_se",
                "perimeter_se",
                "area_se",
                "smoothness_se",
                "compactness_se",
                "concavity_se",
                "concave_points_se",
                "symmetry_se",
                "fractal_dimension_se",
                "radius_worst",
                "texture_worst",
                "perimeter_worst",
                "area_worst",
                "smoothness_worst",
                "compactness_worst",
                "concavity_worst",
                "concave_points_worst",
                "symmetry_worst",
                "fractal_dimension_worst",
                "diagnosis",
            ]
        ]

    def _preprocess_rice(self):
        pass

    def _preprocess_sonar(self):
        pass

    def _preprocess_parkinsons(self):
        self.dataset.df = self.dataset.df[
            [
                "MDVP:Fo(Hz)",
                "MDVP:Fhi(Hz)",
                "MDVP:Flo(Hz)",
                "MDVP:Jitter(%)",
                "MDVP:Jitter(Abs)",
                "MDVP:RAP",
                "MDVP:PPQ",
                "Jitter:DDP",
                "MDVP:Shimmer",
                "MDVP:Shimmer(dB)",
                "Shimmer:APQ3",
                "Shimmer:APQ5",
                "MDVP:APQ",
                "Shimmer:DDA",
                "NHR",
                "HNR",
                "RPDE",
                "DFA",
                "spread1",
                "spread2",
                "D2",
                "PPE",
                "status",
            ]
        ]

    def _preprocess_spam(self):
        pass

    def _preprocess_magic(self):
        pass

    def _preprocess_glass(self):
        self.dataset.df.drop(["id"], axis=1, inplace=True)

    def _preprocess_ionosphere(self):
        pass
    
    def _preprocess_page_blocks(self):
        pass
    
    def _preprocess_waveform(self):
        pass

    def _preprocess_vehicle(self):
        pass

    def _preprocess_image_segmentation(self):
        pass

    def __init__(self, dataset:Dataset):
        self.standardization_scaler = None
        self.dataset = dataset
        self._known_datasets_preprocessor = {
            "wine": self._preprocess_wine,
            "breast_cancer": self._preprocess_breast_cancer,
            "rice": self._preprocess_rice,
            "sonar": self._preprocess_sonar,
            "parkinsons": self._preprocess_parkinsons,
            "spam": self._preprocess_spam,
            "magic": self._preprocess_magic,
            "glass": self._preprocess_glass,
            "ionosphere": self._preprocess_ionosphere,
            "page_blocks": self._preprocess_page_blocks,
            "waveform": self._preprocess_waveform,
            "vehicle": self._preprocess_vehicle,
            "image_segmentation": self._preprocess_image_segmentation,
        }

    def _load_data(self):
        self.dataset.df = pd.read_csv(self.dataset.dataset_path)

    def _split(self, test_size=0.2, random_state=1998):
        (
            self.dataset.X_train,
            self.dataset.X_test,
            self.dataset.y_train,
            self.dataset.y_test,
        ) = train_test_split(
            self.dataset.X,
            self.dataset.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.dataset.y,
        )

    def standardize(self):
        self.standardization_scaler = StandardScaler().fit(self.dataset.X_train)
        self.dataset.X_train = pd.DataFrame(
            self.standardization_scaler.transform(self.dataset.X_train),
            columns=self.dataset.X.columns,
            index=self.dataset.X_train.index,
        )
        self.dataset.X_test = pd.DataFrame(
            self.standardization_scaler.transform(self.dataset.X_test),
            columns=self.dataset.X.columns,
            index=self.dataset.X_test.index,
        )
        return self

    def destandardize(self):
        if self.standardization_scaler is None:
            raise ValueError(
                "Standardization scaler is not initialized. Please call standardize() first."
            )

        self.dataset.X_train = pd.DataFrame(
            self.standardization_scaler.inverse_transform(self.dataset.X_train),
            columns=self.dataset.X.columns,
            index=self.dataset.X_train.index,
        )
        self.dataset.X_test = pd.DataFrame(
            self.standardization_scaler.inverse_transform(self.dataset.X_test),
            columns=self.dataset.X.columns,
            index=self.dataset.X_test.index,
        )
        return self

    def destandardize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.standardization_scaler is None:
            raise ValueError(
                "Standardization scaler is not initialized. Please call standardize() first."
            )
        destandardized_df = pd.DataFrame(
            self.standardization_scaler.inverse_transform(df),
            columns=df.columns,
            index=df.index,
        )
        return destandardized_df

    def destandardize_df_of_ranges(self, df):
        if self.standardization_scaler is None:
            raise ValueError(
                "Standardization scaler is not initialized. Please call standardize() first."
            )
        map_values = df.map if hasattr(df, "map") else df.applymap
        df_min = map_values(lambda x: x[0])
        df_max = map_values(lambda x: x[1])
        destandardized_df_min = pd.DataFrame(
            self.standardization_scaler.inverse_transform(df_min),
            columns=df_min.columns,
            index=df_min.index,
        )
        destandardized_df_max = pd.DataFrame(
            self.standardization_scaler.inverse_transform(df_max),
            columns=df_max.columns,
            index=df_max.index,
        )

        return pd.DataFrame(
            {
                col: list(zip(destandardized_df_min[col], destandardized_df_max[col]))
                for col in destandardized_df_min.columns
            },
            index=destandardized_df_min.index,
        )

    def preprocess(self):
        self._load_data()
        if self.dataset.dataset_name is not None:
            self._known_datasets_preprocessor[self.dataset.dataset_name]()
        self.dataset.y = self.dataset.df[self.dataset.dataset_class_column_name]
        self.dataset.X = self.dataset.df.drop(
            self.dataset.dataset_class_column_name, axis="columns"
        )
        self._split()
        return self
