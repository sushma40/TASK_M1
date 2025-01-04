import streamlit as st
import numpy as np
import pickle

# Load the saved model and PCA transformer
with open('knn_model.pkl', 'rb') as file:
    loaded_knn = pickle.load(file)

with open('pca_transformer.pkl', 'rb') as file:
    loaded_pca = pickle.load(file)

# Function to take input from the user and make a prediction
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_data_pca = loaded_pca.transform(input_data)
    predicted_species = loaded_knn.predict(input_data_pca)
    return predicted_species[0]

# Streamlit App UI
st.set_page_config(page_title="Iris Flower Species Prediction", page_icon="🌸", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        /* Custom background gradient for the entire app */
        .stApp {
            background: linear-gradient(to right, #ffff66, #ff9999);
            font-family: 'Arial', sans-serif;
        }

        /* Navigation bar styles */
        header {
            background-color: #FF8C94 !important;  /* Soft coral pink background for the header section */
            padding: 10px;
        }

        /* Style the elements in the navigation bar */
        header .stMarkdown {
            color: white !important;  /* White text color for navigation */
            font-weight: bold;
            font-size: 20px;
        }

        /* Header container (title section) */
        .header-container {
            background-color: #FF8C94;
            padding: 10px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            margin-bottom: 15px;
        }

        .header-text {
            font-size: 32px;
            font-weight: bold;
            color: white;
            margin: 0;
        }

        /* Styling the input labels */
        label {
            font-size: 18px;
            color: black;
            font-weight: bold;
        }

        /* Prediction result styling */
        .prediction-text {
            font-size: 24px;
            font-weight: bold;
            color: #333333;
            text-align: center;
            margin-top: 20px;
        }

        /* Button styling */
        .stButton button {
            background-color: #FF8C94;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 25px;
            border: none;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease-in-out;
        }

        .stButton button:hover {
            background-color: #CD5C5C;
            transform: scale(1.05);
        }

        /* Input fields side by side */
        .stNumberInput {
            width: 100% !important;
            margin-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

# Display title with custom HTML and CSS
st.markdown(
    '<div class="header-container">'
    '<p class="header-text">Iris Flower Species Prediction</p>'
    '</div>',
    unsafe_allow_html=True
)

# Description text
st.write("Enter the details of the flower to predict its species.")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)

with col2:
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

# Button to trigger prediction
if st.button('Predict'):
    predicted_class = predict_species(sepal_length, sepal_width, petal_length, petal_width)
    species_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
    species_name = species_map.get(predicted_class, 'Unknown')
    st.markdown(f'<p class="prediction-text">Predicted species: {species_name}</p>', unsafe_allow_html=True)
