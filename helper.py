import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model      import Ridge

@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('data/listings-2.csv')
    df_temp = pd.read_csv('data/listings.csv')
    airbnbs = pd.merge(df_temp, df[['id', 'beds', 'bathrooms', 'bedrooms', 'accommodates', 'neighbourhood_cleansed']], on='id', how='left')
    airbnbs = airbnbs[airbnbs['beds'] > 0]
    airbnbs = airbnbs[airbnbs['price'] < 1000]
    airbnbs = airbnbs.dropna(subset=['price', 'beds', 'bathrooms'])

    return airbnbs

def plot_accom_vs_price(df):
    fig = px.box(df, x="accommodates", y="price",
              labels={"accommodates":"# Guests","price":"Price (USD)"})
    fig.update_layout(
        plot_bgcolor='white', # Clean background
        title_x=0.5,
    )
    fig.update_layout(
        title={
            'text': "Price vs. Number of Guests",
            'x': 0.5,           # 0 = far left, 1 = far right
            'xanchor': 'center' # anchor the title around that x position
        }
    )
    return fig

def plot_neighbourhood_distribution(airbnbs):
    n_counts = airbnbs['neighbourhood'].value_counts().reset_index()
    n_counts.columns = ['Neighbourhood', 'Number of Listings']

    fig = px.bar(
        n_counts,
        x='Neighbourhood',
        y='Number of Listings',
    )

    fig.update_layout(
        xaxis_title="Neighbourhood",
        yaxis_title="Number of Listings",
        xaxis_tickangle=-90,  # Rotate the labels
        title_x=0.5,          # Center the title
        plot_bgcolor='white', # Clean background
        bargap=0.2,           # Slight gap between bars
    )
    
    fig.update_layout(
        title={
            'text': "Distribution of Airbnb Listings by Neighbourhood",
            'x': 0.5,           # 0 = far left, 1 = far right
            'xanchor': 'center' # anchor the title around that x position
        }
    )

    fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)', marker_line_width=1.5)

    return fig

def plot_grouped(df):
    df_grouped = (
    df
    .groupby(['neighbourhood_group','accommodates'])['price']
    .mean()
    .round(2)                      # round to 2 decimals
    .reset_index()
    .pivot(index='accommodates', columns='neighbourhood_group', values='price')
    )
    return df_grouped

style = """
    
        body {
        font-family: Arial, sans-serif;
        line-height: 1.6;
        margin: 20px;
        background-color: #f9f9f9;
        }
        h1, h2, h3 {
        color: #333;
        }
        ul {
        list-style-type: disc;
        margin-left: 20px;
        }
        p {
        margin-bottom: 15px;
        }
        code {
        color: green !important;
        background: none !important;
      }
    
    """
intro_html = """
    <body>
    <p>
        This dataset contains Airbnb listings for New York City starting from March 2025. It provides detailed information about the properties available in various neighborhoods across NYC and is especially useful for understanding the dynamics of short-term rental pricing in the city.
    </p>

    <h4>Core Project Question</h2>
    <p>
        <strong>Question:</strong> Given a neighborhood name, number of beds, number of bathrooms, borough name, latitude, longitude, number of guests, what is the predicted nightly price of an Airbnb listing in NYC?
    </p>

    <h4>Why Readers Should Care</h2>
    <p>
        Readers might find this question and dataset particularly interesting because Airbnb prices can vary widely based on location and property attributes. Understanding these variations is valuable for several reasons:
    </p>
    <ul>
        <li><strong>For Hosts:</strong> It can help property owners set competitive yet profitable pricing strategies.</li>
        <li><strong>For Travelers:</strong> It provides insights into what factors impact cost, helping them make informed booking decisions.</li>
        <li><strong>For Researchers and Urban Planners:</strong> It offers a window into how neighborhood characteristics influence market dynamics in one of the world’s most iconic cities.</li>
    </ul>

    <h4>Dataset Details</h2>
    <p>
        <strong>Number of Rows:</strong> The dataset includes 21,272 records, providing a substantial sample for analysis.
    </p>
    <p>
        <strong>Relevant Columns and Their Descriptions:</strong>
    </p>
    <ul>
        <li>
        <strong><code>neighbourhood</code>:</strong> Indicates the neighborhood where the property is located (e.g., Upper East Side, Midtown, Williamsburg).
      </li>
      <li>
        <strong><code>neighbourhood_group</code>:</strong> Indicates the borough where the property is located (e.g., Manhattan, Brookly, Queens).
      </li>
      <li>
        <strong><code>beds</code>:</strong> Represents the total number of beds available in the property, including all sleeping options (not just the number of bedrooms).
      </li>
      <li>
        <strong><code>bathrooms</code>:</strong> Reflects the total number of bathrooms in the property, including half bathrooms.
      </li>
      <li>
        <strong><code>accommodates</code>:</strong> Reflects the total number people that the property can sleep.
      </li>
      <li>
        <strong><code>latitude/longitude</code>:</strong> Coordinates that give you a precise “address” on Earth’s surface, which is crucial for calculating distances between tourist attractions.
      </li>
    </ul>
    </body>
    """
    
data_cleaning_text = """
    #### Data Cleaning and Imputation
    I removed rows with missing values to simplify the analysis and ensure that every record had complete information for key features like neighborhood, beds, bathrooms, accommodates, latitude, and longitude. Even though this reduced the dataset from ~37,000 to ~22,000 rows, the remaining data is still robust for building a reliable prediction model. I opted to drop rather than impute missing values because filling in beds or bathrooms with averages (or zeros) would have required strong—and potentially flawed—assumptions about those listings (e.g. shared rooms or dorm setups) that could bias our estimates.

    Additionally, I excluded listings with nightly rates above $1,000 to mitigate the undue leverage of ultra-luxury outliers on our regression estimates. Even though this step reduced the dataset from 22,000 to 21,300 rows, the remaining data still offers solid coverage of typical price ranges. This filtering also improved model stability and boosted the cross-validated R² by removing extreme values that were disproportionately driving error.
"""

prediction_problem_markdown = """
    #### Prediction type
    This is a **regression** problem, since our goal is to predict a continuous variable (nightly price).


    #### Response variable
    **`price`** (USD per night)  
    I chose `price` because it directly captures the cost outcome of interest for hosts and guests, and is naturally continuous.


    ### Evaluation metric
    I used **Mean Squared Error (MSE)**

    #### Why MSE?
    - It penalizes larger errors more heavily (squaring differences), which helps focus on reducing big misses.  
    - MSE aligns with the loss function used by OLS and Ridge regression.  
    - I prefer MSE over MAE when large deviations are especially costly and I want a differentiable loss.


    #### Time‑of‑prediction
    All features (`neighbourhood`, `beds`, `bathrooms`, `latitude`, `longitude`, `accommodates`) are known at listing creation—no “future” data (reviews, occupancy) leaks into training.  

"""

baseline_text = """
    #### Model
    I trained a **Ridge regression** model with hyperparameter tuning over  α ∈ {0.01, 0.1, 1, 10} via 5-fold `GridSearchCV`.
    
    #### Features & Encodings
    - **Quantitative (2):**  `beds`, `bathrooms` → Scaled using `StandardScaler` 
    - **Nominal (1):** `neighbourhood_group` → One-hot encoded with `OneHotEncoder(drop='first')`
    - **Ordinal:** none.


    #### Performance
    - **Best CV MAE:** \$88.10  
    - **Test MSE:** 16 870.81  
    - **Test RMSE:** √16 870.81 ≈ \$129.87  
    - **Test R²:** 0.248  


    #### Is this “good”?  
    - An **R² of 0.248** means the model explains about 24.8 % of the variance in nightly prices.  
    - An **RMSE of \$129.87** implies typical prediction errors on the order of \$130 per night.  
    - As a **baseline**, it captures some signal (especially via location and size), but leaves substantial room for improvement, particularly on high-error listings.  
"""

final_text = """
    #### Features added
    - **`accommodates`**  
    Number of guests a listing can host. Larger groups often demand more space and amenities, so including capacity helps capture that price premium.  
    - **Spatial feature (distance)**  
    I computed a distance from each listing to a central reference point using `compute_distance(latitude, longitude)`. Proximity to the city center has a strong but non-linear effect on price.  
    - **Polynomial expansion of distance**  
    By tuning the polynomial degree up to 12 (best = 6), the model can flexibly learn complex spatial pricing curves (e.g. diminishing or accelerating effects of distance).  
    - **Log-transform + scaling of numeric features**  
    Applied `log1p` to `beds`, `bathrooms`, and `accommodates` to reduce skew and stabilize variance before standard scaling. This ensures no single feature dominates due to scale or outliers.

    These additions draw on domain knowledge: capacity and location clearly influence price, and their effects aren’t purely linear.


    #### Algorithm & Hyperparameter Tuning
    - **Model:** Ridge regression (linear model with L2 regularization)  
    - **Pipeline:**  
    1. **Preprocessor**  
        - `log1p` → numeric features (`beds`, `bathrooms`, `accommodates`)  
        - `distance_tf` + `PolynomialFeatures` → (`latitude`, `longitude`) → scale  
        - One-hot encode `neighbourhood_cleansed`  
    2. **Ridge**  
    - **GridSearchCV** (5-fold CV, scoring=`neg_mean_squared_error`) over:  
    - `ridge__alpha`: 10^(–2) … 10^(14) → **best: 1.0**  
    - `preprocessor__distance_poly__poly__degree`: 1 … 12 → **best: 6**


    #### Performance Improvement

    | Model                   | Test MSE   | Test RMSE | Test R²  |
    |------------------------:|-----------:|----------:|---------:|
    | Baseline Ridge (α=10)   | 16 870.81  | 129.87    | 0.248    |
    | Final Ridge (α=1.0, d=6)| 12 554.83  | 112.07    | 0.440    |

    - **MSE ↓ by ≈ 4 316 (25.6 %)**, **RMSE ↓ by ≈ 17.8**  
    - **R² ↑ by 0.192**, so the final model explains 44.0 % of variance vs. 24.8 % for baseline.

    **Conclusion:**  
    Adding capacity and spatially-expanded features, along with log transforms and regularization, yields a substantially more accurate model that captures nearly twice as much variance in nightly prices.  
"""

source_text = """

- **Airbnb NYC Listings**  
  Retrieved from [Inside Airbnb: New York City, New York, United States](https://insideairbnb.com/get-the-data/) on April 22, 2025.  

- **Neighborhood coordinates**  
  Pulled from NYC Open Data “Neighborhood Tabulation Areas” API on April 22, 2025.
"""
uni_text = """
     The vast majority of listings are concentrated in central areas like Bedford-Stuyvesant, Upper East Side and Williamsburg, with a long tail of smaller counts spread across outer neighborhoods—highlighting that supply is heavily skewed towards the most popular borough hubs.
"""

biv_text = """
    Median nightly price increases steadily with the number of guests, and larger listings also show much greater price variability—indicating that while bigger spaces command higher rates, their pricing ranges more widely depending on location and amenities.
"""

pt_text = """
    This pivot table shows that as the number of guests increases, average nightly prices rise across all boroughs, but Manhattan listings are consistently the most expensive while outer boroughs like the Bronx and Staten Island remain noticeably cheaper.
"""


# @st.cache_data
# def train_model(params):
#     # build & fit your pipeline
#     return fitted_pipeline
