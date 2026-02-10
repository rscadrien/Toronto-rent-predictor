import joblib
import pandas as pd
import streamlit as st
from Data_preprocessing.Preprocessing_app import preprocessin_app

# Streamlit app for Toronto Rental Price Prediction

st.title("Toronto Rental Price Prediction")

#User  Inputs
Address = st.text_input("Enter the address of the rental property. Example: 6020 Bathurst Street, Toronto, ON, M2R 1Z8 ")
Building_type = st.selectbox("Select Building Type", options=['Apartment', 'House', 'Condo', 'Basement', 'Townhouse', 'Duplex/Triplex'])
bedrooms = st.number_input("Number of Bedrooms", min_value=0, max_value=5, step=1)
bathrooms = st.number_input("Number of bathrooms", min_value=1.0, max_value=3.0, value=1.0, step=0.5)
Size = st.number_input("What is the size (in sqft)?",min_value=0, max_value=2000)
parking = st.number_input("Number of parking", min_value=0, max_value=3, step=1)
furnished = st.selectbox("Furnished?", options=['Yes', 'No'])
AC = st.selectbox("Air conditioning?", options=['Yes', 'No'])
Smoking = st.selectbox("Is smoking permitted?", options=['Yes','Outdoor only','No'])
Pet = st.selectbox("Are pet allowed?", options=['Yes','Limited','No'])
Internet = st.selectbox("Is Wifi included in the rent?",options=['Yes','No'])
TV = st.selectbox("Is cable TV included in the rent?", options=['Yes', 'No'])
Balcony = st.selectbox("Is there a balcony?", options=['Yes','No'])
Yard = st.selectbox('Is there a yard?',options=['Yes','No'])
Hydro = st.selectbox('Is the hydro included in the rent?', options=['Yes','No'])
Heat = st.selectbox('Is the heater included in the rent?', options=['Yes','No'])
Water = st.selectbox('Is water included in the rent?', options=['Yes','No'])
Laundry_un = st.selectbox('Is there laundry in the unit?',options=['Yes','No'])
Laundry_bd = st.selectbox('Is there laundry in the building?', options=['Yes','No'])
Fridge = st.selectbox('Is there a fridge/freezer in the unit?', options=['Yes','No'])
Dishwasher = st.selectbox('Is there a dishwasher in the unit?', options=['Yes','No'])

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
