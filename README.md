# Household Electricity Consumption Prediction

This project was developed during a 1-month Artificial Intelligence and Machine Learning internship at Aplict Solution.

The main objective of the project is to predict household electricity consumption using Machine Learning techniques.

## Project Workflow

### Week 1 - Data Analysis and EDA
- Dataset loading
- Data understanding
- Missing value checking
- Data cleaning
- Exploratory Data Analysis
- Data visualization
- Correlation analysis

### Week 2 - Machine Learning and Prediction
- Feature selection
- Train-test split
- Feature scaling
- Regression model training
- Model prediction
- Model evaluation using MAE, MSE, RMSE, and R² Score

### Week 3 - Model Improvement and Optimization
- Random Forest model improvement
- Hyperparameter tuning
- GridSearchCV
- Cross-validation
- Model comparison
- Feature importance analysis
- Final model selection

### Week 4 - Streamlit Application
- Interactive web application
- Dataset overview
- Electricity consumption prediction
- Model performance section
- Project information page

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook
- Visual Studio Code

## Project Structure

AIML/
- data/
  - electricity.txt
  - electricity_sample.csv
  - electricity_cleaned.csv
  - model_comparison.csv
- notebook/
  - week1.ipynb
  - week2.ipynb
  - week3.ipynb
- app.py
- model.pkl
- scaler.pkl
- requirements.txt
- README.md

## Machine Learning Problem

This project is a supervised Machine Learning regression problem.

The target variable is Global Active Power.

The final application uses a trained Machine Learning model to predict household electricity consumption from user-provided electrical measurements.

## How to Run the Project

Install the required libraries:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

## Developed By

Krutagana Patel

B.Tech Computer Science and Engineering Student