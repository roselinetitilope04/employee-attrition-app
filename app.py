import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("attrition_pipeline.pkl")

st.title("Employee Attrition Prediction App")
st.write("Predict whether an employee is likely to leave the company.")

# ---- User Inputs ----
Age = st.number_input("Age", 18, 60, 30)
BusinessTravel = st.selectbox("Business Travel", [0, 1, 2])
DailyRate = st.number_input("Daily Rate", 100, 1500, 500)
DistanceFromHome = st.number_input("Distance From Home", 1, 50, 10)
Education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
EducationField = st.selectbox("Education Field", [0, 1, 2, 3, 4, 5])
EnvironmentSatisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
HourlyRate = st.number_input("Hourly Rate", 30, 100, 50)
JobInvolvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
JobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5])
JobRole = st.selectbox("Job Role", [0, 1, 2, 3, 4, 5, 6, 7, 8])
JobSatisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 5000)
MonthlyRate = st.number_input("Monthly Rate", 1000, 30000, 15000)
NumCompaniesWorked = st.number_input("Companies Worked", 0, 10, 2)
PercentSalaryHike = st.number_input("Percent Salary Hike", 0, 30, 10)
RelationshipSatisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3])
TotalWorkingYears = st.number_input("Total Working Years", 0, 40, 10)
TrainingTimesLastYear = st.number_input("Training Times Last Year", 0, 10, 3)
WorkLifeBalance = st.selectbox("Work Life Balance", [1, 2, 3, 4])
YearsAtCompany = st.number_input("Years at Company", 0, 40, 5)
YearsInCurrentRole = st.number_input("Years in Current Role", 0, 20, 3)
YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", 0, 15, 2)
YearsWithCurrManager = st.number_input("Years With Current Manager", 0, 20, 3)

# Build dataframe
new_data = pd.DataFrame([{
    "Age": Age,
    "BusinessTravel": BusinessTravel,
    "DailyRate": DailyRate,
    "DistanceFromHome": DistanceFromHome,
    "Education": Education,
    "EducationField": EducationField,
    "EnvironmentSatisfaction": EnvironmentSatisfaction,
    "HourlyRate": HourlyRate,
    "JobInvolvement": JobInvolvement,
    "JobLevel": JobLevel,
    "JobRole": JobRole,
    "JobSatisfaction": JobSatisfaction,
    "MonthlyIncome": MonthlyIncome,
    "MonthlyRate": MonthlyRate,
    "NumCompaniesWorked": NumCompaniesWorked,
    "PercentSalaryHike": PercentSalaryHike,
    "RelationshipSatisfaction": RelationshipSatisfaction,
    "StockOptionLevel": StockOptionLevel,
    "TotalWorkingYears": TotalWorkingYears,
    "TrainingTimesLastYear": TrainingTimesLastYear,
    "WorkLifeBalance": WorkLifeBalance,
    "YearsAtCompany": YearsAtCompany,
    "YearsInCurrentRole": YearsInCurrentRole,
    "YearsSinceLastPromotion": YearsSinceLastPromotion,
    "YearsWithCurrManager": YearsWithCurrManager
}])
# ---- FIX FEATURE NAME MISMATCH (CORRECT WAY) ----

# Model expects these EXACT feature names (including spaces)
model_features = list(model.feature_names_in_)

# Build a dict matching model feature names
aligned_data = {}

for col in model_features:
    clean_col = col.strip()  # match UI column
    aligned_data[col] = new_data.get(clean_col, 0)

# Create DataFrame EXACTLY as model expects
new_data = pd.DataFrame([aligned_data])

# Predict
prediction = model.predict(new_data)[0]
probability = model.predict_proba(new_data)[0][1]

if prediction == 1:
    st.error(f"⚠️ Employee is likely to leave ({probability:.2%})")
else:
    st.success(f"✅ Employee likely to stay ({1 - probability:.2%})")
