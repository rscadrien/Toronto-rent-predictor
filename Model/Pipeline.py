from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

def full_pipeline(df):
    # Define the columns for each type of transformation
    mean_scal_col = ['latitude', 'longitude','distance to downtown (km)', 'distance to Forest Hill (km)',
            'distance to Rosedale (km)', 'distance to Lawrence Park (km)', 'distance to Flemingdon Park (km)',
            'distance to Weston (km)', 'distance to Dorset Park (km)']
    scal_col = ['Bedrooms', 'Bathrooms', 'Parking Included'
           ,'Size (sqft)', 'Rooms']
    one_col = ['Building Type']

    #Pipeline for mean imputation and scaling
    mean_scal_pip = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scal1',  StandardScaler())
    ])
    #Pipeline for scaling
    scal_pip = Pipeline([
        ('scal2',  StandardScaler())
    ])

    #Pipeline for one-hot encoding
    # Pipeline for Building type
    one_pip = Pipeline([
    ('one',  OneHotEncoder())
    ])
    # Combine the pipelines into a ColumnTransformer
    preprocessor = ColumnTransformer([
        ('mean_scal', mean_scal_pip, mean_scal_col),
        ('scal', scal_pip, scal_col),
        ('one', one_pip, one_col)],
        remainder='passthrough'
    )

    #Define the ML model
    model = XGBRegressor(n_estimators = 200, max_depth = 7, learning_rate = 0.05, subsample = 0.8)

    # Combine the preprocessor and model into a full pipeline
    full_pip = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return full_pip
