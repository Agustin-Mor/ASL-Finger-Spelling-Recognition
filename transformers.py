# %%
from pandas import DataFrame

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# %%

class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to wrap LabelEncoder.
    """

    def __init__(self):
        self.encoder = None
        

    def fit(self, X, y=None):

        # Instantiating encoder
        self.encoder = LabelEncoder()

        # Fitting encoder to data
        self.encoder.fit(X)

        return self
    

    def transform(self, X):

        # Check if the encoder has been fitted
        if self.encoder is None:
            raise RuntimeError("You must fit the encoder before transforming data!")
        
        # Transform data 
        encoded_df = DataFrame(self.encoder.transform(X))
        
        return encoded_df
    

    def inverse_transform(self, X):

        # Inverse transform data
        unencoded_df = DataFrame(self.encoder.inverse_transform(X.to_numpy().ravel()))

        return unencoded_df



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