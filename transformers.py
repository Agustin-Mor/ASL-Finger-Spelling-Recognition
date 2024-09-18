# %%
from pandas import DataFrame

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


# %%
class LabelEncoder(BaseEstimator, TransformerMixin):
    """
    One-Hot-Encodes the labels.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = None


    def fit(self, X, y=None):

        # Instantiating the encoder
        self.encoder = OneHotEncoder(feature_name_combiner=(lambda _, x: str(x)), sparse_output=False)

        # Ensuring input is a DataFrame
        df = DataFrame(X)

        # Fitting encoder
        self.encoder.fit(df)

        return self
    

    def transform(self, X):

        # Ensuring input is a DataFrame
        df = DataFrame(X)

        # Encoding features from 'label' column
        encoded_data = self.encoder.transform(df)

        # Turning encoded data into a DataFrame
        encoded_data_df = DataFrame(encoded_data, columns=self.encoder.get_feature_names_out())

        return encoded_data_df

# %%
class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Initializes the FeatureScaler with a specified feature range for scaling.
    
    Parameters:
    feature_range : tuple, default=(0, 1)
        Desired range of transformed data.
    """

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.scaler = None


    def fit(self, df, y=None):

        # Initialize MinMaxScaler with the given feature range
        self.scaler = MinMaxScaler(feature_range=self.feature_range)

        # Fit the scaler to the data (X)
        self.scaler.fit(df)
        
        return self


    def transform(self, df):

        # Check if the scaler has been fitted
        if self.scaler is None:
            raise RuntimeError("You must fit the scaler before transforming data!")
        
        # Transform the numeric columns
        scaled_numeric_data = self.scaler.transform(df)

        # Convert the scaled data back to a DataFrame
        scaled_numeric_df = DataFrame(scaled_numeric_data, columns=df.columns)

        return scaled_numeric_df