# 🏠 Toronto Rental Price Prediction

Predict rental prices in Toronto using machine learning, based on data collected from **Kijiji in January 2023**. The project includes feature engineering, preprocessing, and an **XGBoost regression model**, with a **Streamlit web app** for real-time inference.

---

## 📄 Dataset

- **Source:** Kijiji Canada (January 2023)  
- **Features include:**
  - Address, building type, number of bedrooms and bathrooms  
  - Furnishing, pet policies, smoking, air conditioning  
  - Utilities included (Hydro, Heat, Water)  
  - Appliances (Laundry, Fridge/Freezer, Dishwasher)  
  - Distances to key neighborhoods (Downtown, Forest Hill, Rosedale, Lawrence Park, etc.)  
  - Rental price in CAD

> Note: Rental prices have remained relatively stable since January 2023.

---

## ⚙️ Project Overview

This project has three main components:

1. **Data Preprocessing & Feature Engineering**
   - Clean and impute missing values
   - Encode categorical and binary features
   - Compute distances to neighborhoods using geocoding + Haversine formula
   - Derive new features (total rooms, property size per room, utilities, appliances)

2. **Machine Learning Pipeline**
   - Preprocessing: mean imputation, scaling, one-hot encoding
   - Model: XGBoost Regressor 
   - Combined into a **single scikit-learn Pipeline**

3. **Inference with Streamlit**
   - Users input a rental listing
   - The app preprocesses the input in real time
   - Returns a predicted rental price

---

## 🛠️ Installation & Usage

For the easiest experience, use the interactive web application:
https://toronto-rent-predictor.streamlit.app/

---
## 📄 License
This project is licensed under the MIT License.
---
## 📬 Contact
Questions or suggestions? Reach out at:
adridevolder@hotmail.com
