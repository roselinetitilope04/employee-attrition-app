import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("attrition_pipeline.pkl")

st.title("Employee Attrition Prediction App")
st.write("Enter employee details to predict the likelihood of attrition.")

# ----------- User Inputs -----------
Age = st.number_input("Age", 18, 60, 30)
BusinessTravel = st.selectbox("Business Travel", [0, 1, 2])
DailyRate = st.number_input("Daily Rate", 100, 1500, 500)
DistanceFromHome = st.number_input("Distance From Home", 1, 30, 10)
Education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
EducationField = st.selectbox("Education Field", [0, 1, 2, 3, 4, 5])
EnvironmentSatisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
HourlyRate = st.number_input("Hourly Rate", 30, 100, 60)
JobInvolvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
JobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5])
JobRole = st.selectbox("Job Role", [0, 1, 2, 3, 4, 5, 6, 7, 8])
JobSatisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 5000)
MonthlyRate = st.number_input("Monthly Rate", 2000, 30000, 10000)
NumCompaniesWorked = st.number_input("Number of Companies Worked", 0, 10, 2)
PercentSalaryHike = st.number_input("Percent Salary Hike", 10, 25, 15)
RelationshipSatisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3])
TotalWorkingYears = st.number_input("Total Working Years", 0, 40, 5)
TrainingTimesLastYear = st.number_input("Training Times Last Year", 0, 6, 2)
WorkLifeBalance = st.selectbox("Work Life Balance", [1, 2, 3, 4])
YearsAtCompany = st.number_input("Years at Company", 0, 40, 3)
YearsInCurrentRole = st.number_input("Years in Current Role", 0, 20, 2)
YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", 0, 15, 1)
YearsWithCurrManager = st.number_input("Years With Current Manager", 0, 20, 3)

# ----------- Prediction Button -----------
if st.button("Predict Attrition"):
    input_data = pd.DataFrame([{
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

    # Ensure correct column order
    input_data = input_data[model.feature_names_in_]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Employee is likely to leave (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Employee likely to stay (Probability: {1 - probability:.2%})")
