import numpy as np
import pandas as pd

def withdraw_columns(df, columns_to_remove):
    """
    Remove specified columns from a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    columns_to_remove : list of str
        Column names to drop from the DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the specified columns removed.
    """
    df = df.drop(columns=columns_to_remove)
    return df

def selection_rows(df, columns_to_check, min_price, max_price):
    """
    Filter rows based on missing values and price range.

    Rows with missing values in the specified column are removed, and
    only rows with prices within the given range are retained.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    columns_to_check : str
        Column name used to check for missing values.
    min_price : float
        Minimum allowed price.
    max_price : float
        Maximum allowed price.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame.
    """
    df = df.dropna(subset=[columns_to_check])
    df = df[(df['Price($)'] >= min_price) & (df['Price($)'] <= max_price)]
    return df

def encode_single_categorical(df,column,mapping):
    """
    Encode a categorical column using a provided mapping.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    column : str
        Name of the categorical column to encode.
    mapping : dict
        Dictionary mapping original categories to numerical values.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the encoded column.
    """    
    df[column] = df[column].map(mapping)
    return df

def encode_multiple_columns(df, old_column, new_columns, mapping):
    """
    Encode a single categorical column into multiple numerical columns.

    The original column is replaced by several new columns defined by
    the mapping values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    old_column : str
        Name of the categorical column to encode.
    new_columns : list of str
        Names of the new columns to be created.
    mapping : dict
        Mapping from original values to tuples/lists corresponding
        to the new columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with expanded encoded columns.
    """
    df[new_columns] = df[old_column].map(mapping).apply(pd.Series)
    df = df.drop(columns=[old_column])
    return df

def encode_appliance(df, appliances_list):
    """
    Encode appliance availability into binary columns.

    Each appliance is converted into a binary feature indicating
    whether it is present in the listing.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing an 'Appliances' column.
    appliances_list : list of str
        List of appliances to encode.

    Returns
    -------
    pandas.DataFrame
        DataFrame with appliance columns encoded as binary features.
    """
    df['Appliances'] = df['Appliances'].fillna('')
    for app in appliances_list:
        df[app] = df['Appliances'].apply(lambda x: int(app in x))

    df = df.drop(columns = ['Appliances'])
    return df

def distance(lat1, lon1, lat2, lon2):
    """
    Compute the great-circle distance between two geographic points.

    Uses the Haversine formula to calculate distances on the Earth's surface.

    Parameters
    ----------
    lat1, lon1 : float or array-like
        Latitude and longitude of the first point(s).
    lat2, lon2 : float
        Latitude and longitude of the second point.

    Returns
    -------
    float or numpy.ndarray
        Distance in kilometers.
    """
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 +np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def encode_distance(df, name_column, lat_place, lon_place):
    """
    Compute and encode distance to a reference location.

    Calculates the Haversine distance between each listing and a fixed
    geographic reference point.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing 'latitude' and 'longitude' columns.
    name_column : str
        Name of the new distance column.
    lat_place : float
        Latitude of the reference location.
    lon_place : float
        Longitude of the reference location.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the distance column added.
    """
    df[name_column] = distance(
        df['latitude'].values,
        df['longitude'].values,
        lat_place,
        lon_place
    )
    return df

def new_column_sum(df, new_column, column_1, column_2):
    """
    Create a new column as the sum of two existing columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    new_column : str
        Name of the new column to create.
    column_1 : str
        First column to sum.
    column_2 : str
        Second column to sum.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the new summed column.
    """
    df[new_column] = df[column_1] + df[column_2]
    return df


def encode_size(df, name_column, column_mean):
    """
    Clean and impute property size values.

    Converts size values to numeric format and imputes missing values
    using the mean size grouped by another feature (e.g., number of rooms).

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    name_column : str
        Name of the size column (e.g., 'Size (sqft)').
    column_mean : str
        Column used to compute group-wise means for imputation.

    Returns
    -------
    pandas.DataFrame
        DataFrame with cleaned and imputed size values.
    """
    df[name_column] = df[name_column].replace('Not Available', np.nan)
    df[name_column] = df[name_column].str.replace(',','')
    df[name_column] = df[name_column].astype(float)
    mean_size_per_room = df.groupby(column_mean)[name_column].mean()
    df[name_column] = df.apply(
        lambda row: mean_size_per_room[row[column_mean]] if pd.isna(row[name_column]) else row[name_column], axis=1)
    return df

def feature_engineering_Toronto(df):
    """
    Perform full feature engineering pipeline for Toronto rental data.

    This function cleans the dataset, encodes categorical variables,
    creates derived features, computes geographic distances, and prepares
    the data for machine learning models.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw Toronto rental listings DataFrame.

    Returns
    -------
    pandas.DataFrame
        Fully processed DataFrame ready for modeling.
    """
    # Remove unnecessary columns and filter rows based on price
    column_delete = ['Unnamed: 0','Address',"Title", 'Date Posted', 'Move-In Date', 'Visit Counter', 'url', 'Description',
                     'Amenities', 'Agreement Type']
    df = withdraw_columns(df, column_delete)
    df = selection_rows(df, "Bedrooms", 1000, 4000)
    # Encode number of bedrooms
    Bedroom_mapping ={
        'Bachelor/Studio': 0.5,
        '1': 1,
        '1 + Den': 1.5,
        '2': 2,
        '2 + Den': 2.5,
        '3': 3,
        '3 + Den': 3.5,
        '4': 4,
        '4 + Den': 4.5,
        '5+': 5
    }
    df = encode_single_categorical(df, 'Bedrooms', Bedroom_mapping)

    # Encode number of bathrooms
    Bathroom_mapping ={
        '1': 1,
        '1.5': 1.5,
        '2': 2,
        '2.5': 2.5,
        '3': 3,
    }
    df = encode_single_categorical(df, 'Bathrooms', Bathroom_mapping)

    #Encode parking included
    Parking_mapping ={
        '0': 0,
        '1': 1,
        '2': 2,
        '3+': 3,
    }
    df = encode_single_categorical(df, 'Parking Included', Parking_mapping)

    #Encode furnished or not
    Furnished_mapping ={
        'No': 0,
        'Yes': 1,
    }
    df = encode_single_categorical(df, 'Furnished', Furnished_mapping)

    #Encode air conditioning or not
    Air_mapping ={
        'No': 0,
        'Not Available': 0,
        'Yes': 1,
    }
    df = encode_single_categorical(df, 'Air Conditioning', Air_mapping)

    #Encode smoking allowed or not
    Smoking_mapping ={
        'No': 0,
        'Outdoors only': 0.5,
        'Yes': 1,
    }
    df = encode_single_categorical(df, 'Smoking Permitted', Smoking_mapping)

    #Encode pet friendly or not
    Pet_mapping ={
        'No': 0,
        'Limited': 0.5,
        'Yes': 1,
    }
    df = encode_single_categorical(df, 'Pet Friendly', Pet_mapping)

    #Create new column for total rooms
    df = new_column_sum(df, 'Rooms', 'Bedrooms', 'Bathrooms')

    #Encoding Size (sqft)
    df = encode_size(df, 'Size (sqft)', 'Rooms')

    #Encode Wifi and Cable TV
    wifi_mapping = {
        'Not Included': (0,0),
        'Internet': (1,0),
        'Cable / TVInternet': (1,1),
        'Cable / TV': (0,1)
    }
    old_column = 'Wi-Fi and More'
    new_column = ['Internet', 'Cable_TV']
    df = encode_multiple_columns(df, old_column, new_column, wifi_mapping)

    #Encode personal outdoor space
    outdoor_mapping = {
        'Not Included': (0,0),
        'Balcony': (1,0),
        'Yard': (0,1),
        'YardBalcony': (1,1)
    }
    old_column = 'Personal Outdoor Space'
    new_column = ['Balcony', 'Yard']
    df = encode_multiple_columns(df, old_column, new_column, outdoor_mapping)

    #Encode utilities included
    utilities_mapping = {
        'Hydro_No,Heat_Yes,Water_Yes': (0,1,1),
        'Hydro_Yes,Heat_Yes,Water_Yes': (1,1,1),
        'Hydro_No,Heat_Yes,Water_No': (0,1,0),
        'Hydro_No,Heat_No,Water_Yes': (0,0,1),
        'Hydro_Yes,Heat_No,Water_Yes': (1,0,1),
        'Hydro_Yes,Heat_No,Water_No': (1,0,0),
        'Hydro_Yes,Heat_Yes,Water_No': (1,1,0),
        'NaN': (0,0,0)
    }
    old_column = 'Utilities'
    new_column = ['Hydro', 'Heat', 'Water']
    df = encode_multiple_columns(df, old_column, new_column, utilities_mapping)

    #Encode appliances
    appliances_list = ['Laundry (In Building)', 'Laundry (In Unit)', 'Fridge / Freezer', 'Dishwasher']
    df = encode_appliance(df, appliances_list)

    #Encode distance to downtown
    # Position of the CN Tower
    lat_CN = 43.6426
    lon_CN = -79.3871
    df = encode_distance(df, 'distance to downtown (km)', lat_CN, lon_CN)

    #Encode distance to Forest Hill
    lat_forest = 43.6936
    lon_forest = -79.4139
    df = encode_distance(df, 'distance to Forest Hill (km)', lat_forest, lon_forest)

    #Encode distance to Rosedale
    lat_rose = 43.6790
    lon_rose = -79.3780
    df = encode_distance(df, 'distance to Rosedale (km)', lat_rose, lon_rose)

    #Encode distance to Lawrence Park
    lat_law = 43.7220
    lon_law = -79.3879
    df = encode_distance(df, 'distance to Lawrence Park (km)', lat_law, lon_law)

    #Encode distance to Flemingdon Park
    lat_flem = 43.7184
    lon_flem = -79.3332
    df = encode_distance(df, 'distance to Flemingdon Park (km)', lat_flem, lon_flem)

    #Encode distance to Weston
    lat_west = 43.7007
    lon_west = -79.5138
    df = encode_distance(df, 'distance to Weston (km)', lat_west, lon_west)

    #Encode distance to Dorset Park
    lat_dor = 43.7612
    lon_dor = -79.2846
    df = encode_distance(df, 'distance to Dorset Park (km)', lat_dor, lon_dor)

    return df

