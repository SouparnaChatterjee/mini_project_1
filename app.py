import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import warnings
warnings.filterwarnings("ignore")
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to load data from a URL
@st.cache
def load_data(url):
    try:
        logger.info(f"Attempting to load data from {url}")
        data = pd.read_csv(url)
        logger.info("Data loaded successfully")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"Error loading data: {e}")
        return None

# URL for the CSV file
url = "https://drive.google.com/uc?id=1z29oToTgvuK9QsJHs_9WcTUvzuZeXe0L"

# Load the dataset
data = load_data(url)

if data is not None:
    # Function to generate histograms
    def generate_histograms():
        plt.figure(figsize=(20, 25), facecolor='pink')
        for i, column in enumerate(data.columns[:9], 1):
            plt.subplot(3, 3, i)
            sns.histplot(data[column], kde=True)
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
        plt.suptitle('Histograms of First 9 Columns', fontsize=20, y=0.93)
        plt.tight_layout()
        return plt.gcf()

    # Function to generate scatter plot
    def generate_scatter_plot():
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=data, x='MinTemp', y='MaxTemp', hue='Rainfall', size='Rainfall', sizes=(20, 200))
        plt.title('Minimum Temperature vs Maximum Temperature by Rainfall')
        plt.xlabel('Minimum Temperature (°C)')
        plt.ylabel('Maximum Temperature (°C)')
        plt.colorbar(label='Rainfall (mm)')
        return plt.gcf()

    # Function to handle missing values
    def handle_missing_values():
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        for col in categorical_cols:
            data[col] = data[col].fillna(data[col].mode()[0])

    # Generate visualizations and handle missing values
    handle_missing_values()

    # Streamlit app
    st.title('Weather Data Analysis')

    st.header('First 5 Rows of the Dataset')
    st.dataframe(data.head())

    st.header('Shape of the Dataset')
    st.write(data.shape)

    st.header('Statistical Description of the Data')
    st.dataframe(data.describe())

    st.header('Null Values in Each Column')
    st.dataframe(data.isnull().sum().to_frame())

    st.header('Visualizations')

    st.subheader('Histograms of First 9 Columns')
    st.pyplot(generate_histograms())

    st.subheader('Minimum Temperature vs Maximum Temperature by Rainfall')
    st.pyplot(generate_scatter_plot())
else:
    st.warning("Data could not be loaded. Please check the data URL and try again.")
