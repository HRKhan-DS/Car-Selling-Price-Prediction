import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import pickle

# Load the pre-trained Gradient Boosting model with pickle
with open("GradientBoosting_model.pkl", "rb") as model_file:
    gb_model = pickle.load(model_file)

# Load the cleaned data
cleaned_data = pd.read_csv('Cleaned_data.csv')

def main():
    st.title('Car Selling Price Prediction')

    # Sidebar with user inputs
    st.sidebar.header('Enter Car Details:')
    car_name = st.sidebar.selectbox('Select Car Model', cleaned_data['Car_Name'].unique())
    year = st.sidebar.number_input('Manufacturing Year (e.g., 2013)', min_value=2003, max_value=2025)
    present_price = st.sidebar.number_input('Present Price (in Lac)', min_value=0.0, step=0.01)
    kms_driven = st.sidebar.number_input('Kilometers Driven', min_value=0, step=100)
    fuel_type = st.sidebar.selectbox('Fuel Type (Petrol/Diesel/CNG)', cleaned_data['Fuel_Type'].unique())
    seller_type = st.sidebar.selectbox('Seller Type (Dealer/Individual)', cleaned_data['Seller_Type'].unique())
    transmission = st.sidebar.selectbox('Transmission Type (Manual/Automatic)', cleaned_data['Transmission'].unique())
    owner = st.sidebar.number_input('Number of Previous Owners', min_value=0, max_value=2)

    # Validate inputs
    if st.sidebar.button('Predict Price'):
        if present_price == 0 or kms_driven == 0:
            st.error("Please enter valid values for Present Price and Kilometers Driven.")
        else:
            # Create a new DataFrame with user input
            user_input = pd.DataFrame({
                'Car_Name': [car_name],
                'Year': [year],
                'Present_Price': [present_price],
                'Kms_Driven': [kms_driven],
                'Fuel_Type': [fuel_type],
                'Seller_Type': [seller_type],
                'Transmission': [transmission],
                'Owner': [owner]
            })

            # Make predictions
            predicted_price = gb_model.predict(user_input)

            # Display the predicted selling price in the main content area
            st.subheader('Predicted Selling Price:')
            st.write(f"${predicted_price[0]:,.2f} Lac")

    # Display instructions or descriptions
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.write("This is a simple Car Selling Price Prediction.")
    st.write("Please select the car model, manufacturing year, and enter other details on the left sidebar.")
    st.write("Click the 'Predict' button to see the estimated selling price based on the entered details.")

if __name__ == "__main__":
    main()
