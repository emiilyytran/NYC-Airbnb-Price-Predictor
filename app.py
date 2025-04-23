import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
# from sklearn.linear_model      import Ridge
from helper import *

st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    layout="wide",
)

st.title("Empire Estimator: Predicting NYC Airbnb Rates 🏠")
st.markdown("""By: Emily Tran  
            April 22, 2025""")
st.markdown("## Introduction")
st.markdown("---")
st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

# --- render HTML intro ---
st.markdown(intro_html, unsafe_allow_html=True)

# --- STEP 2 ---
st.markdown("## Data Cleaning and Exploratory Data Analysis")
st.markdown("---")
st.markdown(data_cleaning_text)
df = load_and_clean_data()
st.dataframe(df.head(10))

st.markdown("#### Univariate Analysis")
st.plotly_chart(plot_neighbourhood_distribution(df))
st.markdown(uni_text)

st.markdown("#### Bivariate Analysis")
st.plotly_chart(plot_accom_vs_price(df))
st.markdown(biv_text)

st.markdown("#### Interesting Aggregate")
st.write("##### Avg. Price per #Guests by Neighbourhood Group")
st.dataframe(plot_grouped(df))
st.markdown(pt_text)

# --- STEP 3 ---
st.markdown("## Framing a Prediction Problem")
st.markdown("---")
st.markdown(prediction_problem_markdown)

# --- STEP 4 ---
st.markdown("## Baseline Model")
st.markdown("---")
st.markdown(baseline_text)

# --- STEP 5 ---
st.markdown("## Final Model")
st.markdown("---")
st.markdown(final_text)

#  --- END ---
st.markdown("---")
st.markdown("#### Data Sources")
st.markdown(source_text)


