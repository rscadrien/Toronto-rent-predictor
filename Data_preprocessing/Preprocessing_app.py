from geopy.geocoders import Photon
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time
from Data_preprocessing.feature_engineering import encode_distance, encode_single_categorical, new_column_sum, withdraw_columns

def geocode_address(address, sleep=0.3):
    """
    Convert a textual address into geographic coordinates.

    Uses the Photon geocoding service to retrieve latitude and longitude
    from a given address string. Includes a delay between requests to
    respect rate limits and handles common geocoding timeouts.

    Parameters
    ----------
    address : str
        Address to geocode.
    sleep : float, optional
        Time in seconds to wait after each request (default is 0.3).

    Returns
    -------
    tuple of (float or None, float or None)
        Latitude and longitude of the address. Returns (None, None) if
        geocoding fails.
    """
    try:
        geolocator = Photon(user_agent="adrien_geocoder", timeout=10)
        location = geolocator.geocode(address)
        time.sleep(sleep) 
        if location:
            return location.latitude, location.longitude
    except (GeocoderTimedOut, GeocoderUnavailable):
        time.sleep(2)
    return None, None

def preprocessin_app(df):
    """
    Preprocess a single rental listing for inference in the Streamlit app.

    This function performs lightweight feature engineering suitable for
    real-time prediction, including:
    - Geocoding the address
    - Computing distances to key Toronto neighborhoods
    - Encoding binary and categorical features
    - Creating derived numerical features

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a single rental listing.

    Returns
    -------
    pandas.DataFrame
        Preprocessed DataFrame ready for model inference.
    """
    # Preprocessing steps for the inference in the streamlit app
    # Calculating the lattitude and the longitude from the adress
    df[["latitude", "longitude"]] = geocode_address(df["Address"])
    df = withdraw_columns(df,'Address')
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

    # Encode Furnished or not
    Binary_mapping ={
        'No': 0,
        'Yes': 1,
    }

    df = encode_single_categorical(df, 'Furnished', Binary_mapping)

    #Encode air conditioning or not
    df = encode_single_categorical(df, 'Air Conditioning', Binary_mapping)

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

    # Encode Internet
    df = encode_single_categorical(df, 'Internet', Binary_mapping)

    #Encode Cable TV
    df = encode_single_categorical(df, 'Cable_TV', Binary_mapping)

    #Encode Balcony
    df = encode_single_categorical(df, 'Balcony', Binary_mapping)

    #Encode Yard
    df = encode_single_categorical(df, 'Yard', Binary_mapping)

    #Encode Hydro
    df = encode_single_categorical(df, 'Hydro', Binary_mapping)

    #Encode Heat
    df = encode_single_categorical(df, 'Heat', Binary_mapping)

    #Encode Water
    df = encode_single_categorical(df, 'Water', Binary_mapping)

    #Encode Laundry in unit
    df = encode_single_categorical(df, 'Laundry (In Unit)', Binary_mapping)

    #Encode laundry in building
    df = encode_single_categorical(df, 'Laundry (In Building)', Binary_mapping)

    #Encode Fridge
    df = encode_single_categorical(df, 'Fridge / Freezer', Binary_mapping)

    #Encode Dishwasher
    df = encode_single_categorical(df, 'Dishwasher', Binary_mapping)

    return df