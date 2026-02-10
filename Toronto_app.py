import joblib
import pandas as pd
import streamlit as st
from Data_preprocessing.Preprocessing_app import preprocessin_app

# Streamlit app for Toronto Rental Price Prediction

st.title("Toronto Rental Price Prediction by ML/AI")

#User  Inputs
Address = st.text_input("Property address",
                        placeholder="e.g. 6020 Bathurst St, Toronto, ON",
                        help="Full address helps estimate neighborhood pricing more accurately.")
Building_type = st.selectbox("Type of property", options=['Apartment', 'House', 'Condo', 'Basement', 'Townhouse', 'Duplex/Triplex'])
bedrooms = st.number_input("Number of bedrooms", min_value=0, max_value=5, step=1)
bathrooms = st.number_input("Number of bathrooms", min_value=1.0, max_value=3.0, value=1.0, step=0.5)
Size = st.number_input("Living area (sq ft)", min_value=0, max_value=10000,
                       help="Approximate size is fine")
parking = st.number_input("Number of parking spaces", min_value=0, max_value=3, step=1)
furnished = st.selectbox("Furnished?", options=['Yes', 'No'])
AC = st.selectbox("Air conditioning?", options=['Yes', 'No'])
Smoking = st.selectbox("Smoking permitted?", options=['Yes','Outdoor only','No'])
Pet = st.selectbox("Pets allowed?", options=['Yes','Limited','No'])
Internet = st.selectbox("Wifi included in the rent?",options=['Yes','No'])
TV = st.selectbox("Cable TV included in the rent?", options=['Yes', 'No'])
Balcony = st.selectbox("Balcony?", options=['Yes','No'])
Yard = st.selectbox('Yard or outdoor space?',options=['Yes','No'])
Hydro = st.selectbox('Hydro included in the rent?', options=['Yes','No'])
Heat = st.selectbox('Heater included in the rent?', options=['Yes','No'])
Water = st.selectbox('Water included in the rent?', options=['Yes','No'])
Laundry_un = st.selectbox('Laundry in the unit?',options=['Yes','No'])
Laundry_bd = st.selectbox('Laundry in the building?', options=['Yes','No'])
Fridge = st.selectbox('Fridge/freezer in the unit?', options=['Yes','No'])
Dishwasher = st.selectbox('Dishwasher in the unit?', options=['Yes','No'])

X = pd.DataFrame([{
    'Address': Address,
    'Building Type': Building_type,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Size (sqft)': Size,
    'Parking Included': parking,
    'Furnished': furnished,
    'Air Conditioning': AC,
    'Smoking Permitted': Smoking,
    'Pet Friendly': Pet,
    'Internet': Internet,
    'Cable_TV': TV,
    'Balcony': Balcony,
    'Yard': Yard,
    'Hydro': Hydro,
    'Heat': Heat,
    'Water': Water,
    'Laundry (In Unit)': Laundry_un,
    'Laundry (In Building)': Laundry_bd,
    'Fridge / Freezer': Fridge,
    'Dishwasher': Dishwasher
}])


# Prediction Button
if st.button("Predict rent"):
    #Preprocessing steps
    X = preprocessin_app(X)
    # Load model
    model = joblib.load('./Model/toronto_rental_model.pkl')
    price_pred = model.predict(X)

    #Display the rent
    st.subheader('Predicted rent:')
    st.write(f"The predicted rent is {price_pred}")
