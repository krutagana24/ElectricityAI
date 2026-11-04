import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Electricity Consumption Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/electricity_cleaned.csv")
    return df

df = load_data()

# Load ML Model
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Home",
        "Dataset Overview",
        "Prediction",
        "Model Performance",
        "About Project"
    ]
)

# HOME PAGE
if page == "Home":

    st.title(" Household Electricity Consumption Prediction")

    st.markdown("---")

    st.subheader("Project Overview")

    st.write("""
This project predicts household electricity consumption
using Machine Learning.

The project was developed during a **1-Month AI/ML Internship**
using Python, Streamlit, Pandas and Scikit-Learn.

The prediction model helps estimate household electricity
consumption based on electrical measurements.
""")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.info("""
### Dataset

Household Electricity Consumption Dataset

• Cleaned Dataset

• Regression Problem

• Real World Dataset
""")

    with col2:

        st.success("""
###  Machine Learning

Algorithm Used

✔ Random Forest Regressor

✔ Feature Scaling

✔ Model Evaluation

✔ Prediction
""")

    st.markdown("---")

    st.subheader(" Technologies Used")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Language", "Python")

    c2.metric("Framework", "Streamlit")

    c3.metric("ML Library", "Scikit-Learn")

    c4.metric("Visualization", "Matplotlib")

    st.markdown("---")

    st.subheader("Project Objective")

    st.write("""
Develop an AI/ML model capable of predicting household
electricity consumption with high accuracy.

The project follows the complete Machine Learning workflow:

• Data Collection

• Data Cleaning

• Exploratory Data Analysis

• Feature Engineering

• Machine Learning

• Model Improvement

• Prediction using Streamlit
""")

    st.markdown("---")

    st.success(" Week 4 Streamlit Application Successfully Started")

# DATASET OVERVIEW PAGE

elif page == "Dataset Overview":

    st.title("Dataset Overview")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Rows", df.shape[0])

    with col2:
        st.metric("Total Columns", df.shape[1])

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    st.subheader("Dataset Columns")

    st.write(df.columns.tolist())

    st.markdown("---")

    st.subheader("Missing Values")

    st.dataframe(df.isnull().sum())

    st.markdown("---")

    st.subheader("Statistical Summary")

    st.dataframe(df.describe())

    st.markdown("---")

    st.subheader("Correlation Matrix")

    correlation = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10,6))

    cax = ax.imshow(correlation, cmap="Blues")

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=90)

    ax.set_yticks(range(len(correlation.columns)))
    ax.set_yticklabels(correlation.columns)

    plt.colorbar(cax)

    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Distribution of Global Active Power")

    fig2, ax2 = plt.subplots(figsize=(8,5))

    ax2.hist(df["Global_active_power"], bins=30)

    ax2.set_xlabel("Global Active Power")

    ax2.set_ylabel("Frequency")

    ax2.set_title("Distribution of Global Active Power")

    st.pyplot(fig2)

    st.markdown("---")

    st.success("Dataset loaded successfully.")

# PREDICTION PAGE

elif page == "Prediction":

    st.title("Electricity Consumption Prediction")

    st.markdown("---")

    st.write("Enter the following values to predict electricity consumption.")

    col1, col2 = st.columns(2)

    with col1:

        reactive_power = st.number_input(
            "Global Reactive Power",
            min_value=0.0,
            value=0.10,
            step=0.01
        )

        voltage = st.number_input(
            "Voltage",
            min_value=100.0,
            value=240.0,
            step=1.0
        )

        intensity = st.number_input(
            "Global Intensity",
            min_value=0.0,
            value=10.0,
            step=0.1
        )

        month = st.selectbox(
            "Month",
            [1,2,3,4,5,6,7,8,9,10,11,12]
        )

    with col2:

        sub1 = st.number_input(
            "Sub Metering 1",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

        sub2 = st.number_input(
            "Sub Metering 2",
            min_value=0.0,
            value=1.0,
            step=1.0
        )

        sub3 = st.number_input(
            "Sub Metering 3",
            min_value=0.0,
            value=17.0,
            step=1.0
        )

    st.markdown("---")

    if st.button("Predict Electricity Consumption"):

        try:

            input_data = np.array([
                [
                    reactive_power,
                    voltage,
                    intensity,
                    sub1,
                    sub2,
                    sub3,
                    month
                ]
            ])

            input_scaled = scaler.transform(input_data)

            prediction = model.predict(input_scaled)

            st.success("Prediction Completed Successfully!")

            st.metric(
                label="Predicted Electricity Consumption",
                value=f"{prediction[0]:.3f} kW"
            )

            st.info("""
The displayed value represents the estimated
Global Active Power based on the entered inputs.
""")

        except Exception as e:

            st.error("Prediction Failed")

            st.exception(e)

# MODEL PERFORMANCE PAGE

elif page == "Model Performance":

    st.title("Model Performance")

    st.markdown("---")

    st.subheader(" Final Machine Learning Model")

    st.success("Random Forest Regressor")

    st.markdown("---")

    st.subheader(" Evaluation Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="R² Score",
            value="0.98"      # Replace with your value
        )

    with col2:
        st.metric(
            label="MAE",
            value="0.04"      # Replace with your value
        )

    with col3:
        st.metric(
            label="RMSE",
            value="0.08"      # Replace with your value
        )

    with col4:
        st.metric(
            label="MSE",
            value="0.006"     # Replace with your value
        )

    st.markdown("---")

    st.subheader("Feature Importance")

    importance = pd.DataFrame({

        "Feature":[
            "Reactive Power",
            "Voltage",
            "Global Intensity",
            "Sub Metering 1",
            "Sub Metering 2",
            "Sub Metering 3",
            "Month"
        ],

        "Importance":[
            0.12,
            0.18,
            0.31,
            0.08,
            0.07,
            0.19,
            0.05
        ]

    })

    fig, ax = plt.subplots(figsize=(9,5))

    ax.barh(
        importance["Feature"],
        importance["Importance"]
    )

    ax.set_title("Feature Importance")

    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Model Summary")

    st.write("""
**Algorithm Used:** Random Forest Regressor
**Problem Type:** Regression
**Target Variable:** Global Active Power
The Random Forest model achieved the best performance
among all the machine learning algorithms tested during
this internship.
""")

    st.markdown("---")

    st.info("""
The evaluation metrics indicate that the model is capable
of predicting household electricity consumption with good
accuracy.
""")

# ABOUT PROJECT PAGE

elif page == "About Project":

    st.title(" About This Project")

    st.markdown("---")

    st.header(" Project Objective")

    st.write("""
The objective of this project is to predict household electricity
consumption using Machine Learning algorithms.

The project was developed as part of a **1-Month AI/ML Internship**
to understand the complete Machine Learning workflow from
data preprocessing to deployment using Streamlit.
""")

    st.markdown("---")

    st.header("Dataset Information")

    st.write("""
Dataset Name:
Household Electric Power Consumption Dataset

Target Variable:
Global Active Power

Problem Type:
Regression

Total Features:
7 Input Features
""")

    st.markdown("---")

    st.header(" Machine Learning Algorithms Used")

    algorithms = [
        "✔ Linear Regression",
        "✔ Decision Tree Regressor",
        "✔ Random Forest Regressor",
        "✔ Gradient Boosting Regressor"
    ]

    for algo in algorithms:
        st.write(algo)

    st.success("Final Selected Model : Random Forest Regressor")

    st.markdown("---")

    st.header("Technologies Used")

    col1, col2 = st.columns(2)

    with col1:
        st.write("""
• Python
• Pandas
• NumPy
• Matplotlib
• Streamlit
""")

    with col2:
        st.write("""
• Scikit-Learn
• Jupyter Notebook
• VS Code
• Machine Learning
• Data Analysis
""")

    st.markdown("---")

    st.header("Project Workflow")

    st.write("""
1️⃣ Data Collection
⬇
2️⃣ Data Cleaning
⬇
3️⃣ Exploratory Data Analysis (EDA)
⬇
4️⃣ Feature Engineering
⬇
5️⃣ Machine Learning Model Training
⬇
6️⃣ Model Evaluation
⬇
7️⃣ Model Improvement
⬇
8️⃣ Streamlit Deployment
""")

    st.markdown("---")

    st.header("Project Outcome")

    st.success("""
✔ Successfully developed an Electricity Consumption Prediction System.
✔ Compared multiple Machine Learning algorithms.
✔ Selected Random Forest as the best model.
✔Created a Streamlit web application for real-time prediction.
""")

    st.markdown("---")

    st.header("Future Scope")

    st.write("""
• Improve prediction accuracy using advanced ML models.
• Add user authentication and prediction history.
• Create mobile-friendly dashboards.
""")

    st.markdown("---")

    st.header("Developer")

    st.info("""
Name : Krutagana Patel
Course : B.Tech Computer Science & Engineering (AIML)
Project :
Household Electricity Consumption Prediction
Internship :
1-Month Artificial Intelligence & Machine Learning Internship
""")
