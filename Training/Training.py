import  pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Data_preprocessing.feature_engineering import feature_engineering_Toronto
from Model.Pipeline import full_pipeline
from sklearn.metrics import r2_score
import joblib

# Load the dataset
df = pd.read_csv('./Data/Toronto_rental_location.csv')
# Perform feature engineering
df = feature_engineering_Toronto(df)
# Define the target variable and features
X = df.drop('Price($)', axis=1)
y = df['Price($)']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create the full pipeline
pipeline = full_pipeline(df)
# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)
# Evaluate the model on the test set
y_pred_test = pipeline.predict(X_test)
r2_test = r2_score(y_test, y_pred_test)
print(f"R² score on test set: {r2_test:.4f}")
mre_test = np.mean(np.abs((y_test - y_pred_test) / y_test))
print(f"Mean Relative Error on test set: {100*(mre_test):.2f}%")
print(f"1-Mean Relative Error on test set : {100*(1-mre_test):.2f}%")
# Save the trained pipeline
joblib.dump(pipeline, './Model/toronto_rental_model.pkl')




