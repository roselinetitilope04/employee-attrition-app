import streamlit as st
import pandas as pd
import joblib
import time

# ---- Load trained model ----
model = joblib.load("attrition_pipeline.pkl")

# ---- App title ----
st.title("Employee Attrition Prediction App")
st.write("Predict whether an employee is likely to leave the company.")

# ---- User Inputs ----
Age = st.number_input("Age", 18, 60, 30)
BusinessTravel = st.selectbox("Business Travel (0=Rarely, 1=Frequently, 2=Non-Travel)", [0, 1, 2])
DailyRate = st.number_input("Daily Rate", 100, 1500, 500)
DistanceFromHome = st.number_input("Distance From Home (km)", 1, 50, 10)
Education = st.selectbox("Education Level (1-5)", [1, 2, 3, 4, 5])
EducationField = st.selectbox("Education Field (0-5)", [0, 1, 2, 3, 4, 5])
EnvironmentSatisfaction = st.selectbox("Environment Satisfaction (1-4)", [1, 2, 3, 4])
HourlyRate = st.number_input("Hourly Rate", 30, 100, 50)
JobInvolvement = st.selectbox("Job Involvement (1-4)", [1, 2, 3, 4])
JobLevel = st.selectbox("Job Level (1-5)", [1, 2, 3, 4, 5])
JobRole = st.selectbox("Job Role (0-8)", list(range(9)))
JobSatisfaction = st.selectbox("Job Satisfaction (1-4)", [1, 2, 3, 4])
MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 5000)
MonthlyRate = st.number_input("Monthly Rate", 1000, 30000, 15000)
NumCompaniesWorked = st.number_input("Companies Worked", 0, 10, 2)
PercentSalaryHike = st.number_input("Percent Salary Hike (%)", 0, 30, 10)
RelationshipSatisfaction = st.selectbox("Relationship Satisfaction (1-4)", [1, 2, 3, 4])
StockOptionLevel = st.selectbox("Stock Option Level (0-3)", [0, 1, 2, 3])
TotalWorkingYears = st.number_input("Total Working Years", 0, 40, 10)
TrainingTimesLastYear = st.number_input("Training Times Last Year", 0, 10, 3)
WorkLifeBalance = st.selectbox("Work Life Balance (1-4)", [1, 2, 3, 4])
YearsAtCompany = st.number_input("Years at Company", 0, 40, 5)
YearsInCurrentRole = st.number_input("Years in Current Role", 0, 20, 3)
YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", 0, 15, 2)
YearsWithCurrManager = st.number_input("Years With Current Manager", 0, 20, 3)

# ---- Predict Button ----
if st.button("Predict"):
    # ---- Build DataFrame from user input ----
    user_input = {
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
    }

    new_data = pd.DataFrame([user_input])

    # ---- Align DataFrame columns to model features ----
    col_mapping = {col.strip(): col for col in model.feature_names_in_}
    new_data.rename(columns=col_mapping, inplace=True)

    for col in model.feature_names_in_:
        if col not in new_data.columns:
            new_data[col] = 0

    new_data = new_data[model.feature_names_in_]

    # ---- Predict ----
    prediction = model.predict(new_data)[0]
    probability = model.predict_proba(new_data)[0][1]

    # ---- Display Result with Animated Colored Progress Bar ----
    st.write("### Prediction Result")
    bar_placeholder = st.empty()  # placeholder for animation

    if prediction == 1:
        st.error("⚠️ Employee is likely to leave")
        # Animate red progress bar
        for i in range(int(probability*100)+1):
            bar_placeholder.markdown(
                f"""
                <div style="background-color:#ddd; border-radius:5px; width:100%; height:25px;">
                    <div style="width:{i}%; background-color:#FF4B4B; height:100%; border-radius:5px;"></div>
                </div>
                <p>Attrition Probability: {i:.0f}%</p>
                """, unsafe_allow_html=True
            )
            time.sleep(0.01)
    else:
        st.success("✅ Employee likely to stay")
        # Animate green progress bar
        for i in range(int((1-probability)*100)+1):
            bar_placeholder.markdown(
                f"""
                <div style="background-color:#ddd; border-radius:5px; width:100%; height:25px;">
                    <div style="width:{i}%; background-color:#4CAF50; height:100%; border-radius:5px;"></div>
                </div>
                <p>Retention Probability: {i:.0f}%</p>
                """, unsafe_allow_html=True
            )
            time.sleep(0.01)
