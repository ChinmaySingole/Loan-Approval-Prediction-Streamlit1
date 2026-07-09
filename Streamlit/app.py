import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #eef3ff 0%, #f8fbff 45%, #ffffff 100%); }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1300px; }
    h1, h2, h3, h4 { color: #17324d; font-weight: 800; letter-spacing: -0.02em; }
    .hero { background: rgba(255,255,255,0.82); border: 1px solid rgba(255,255,255,0.75); border-radius: 24px; padding: 1.4rem 1.6rem; box-shadow: 0 16px 35px rgba(15, 23, 42, 0.08); margin-bottom: 1rem; }
    .hero-title { font-size: 2rem; font-weight: 900; margin-bottom: 0.2rem; color: #102a43; }
    .hero-subtitle { color: #5f6c7b; font-size: 0.95rem; margin-top: 0.35rem; }
    .pill-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.8rem; }
    .pill { padding: 0.38rem 0.75rem; border-radius: 999px; font-size: 0.8rem; font-weight: 700; border: 1px solid rgba(0,0,0,0.04); }
    .p1 { background: #dbeafe; color: #1d4ed8; }
    .p2 { background: #dcfce7; color: #15803d; }
    .p3 { background: #ffedd5; color: #c2410c; }
    .p4 { background: #ede9fe; color: #6d28d9; }
    .card { background: rgba(255,255,255,0.88); border: 1px solid #e6edf5; border-radius: 20px; padding: 1.1rem 1.2rem; box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06); margin-bottom: 1rem; transition: transform 0.22s ease, box-shadow 0.22s ease; }
    .card:hover { transform: translateY(-2px); box-shadow: 0 14px 34px rgba(15, 23, 42, 0.10); }
    .metric-card { border-radius: 18px; border: 1px solid #e5eaf2; background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%); padding: 1rem; box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05); }
    div[data-testid="stMetric"] { background: transparent; border: none; padding: 0; }
    div[data-testid="stMetricLabel"] { color: #5f6c7b; font-weight: 700; }
    div[data-testid="stMetricValue"] { color: #102a43; font-weight: 900; font-size: 1.5rem; }
    .result-approved, .result-rejected { border-radius: 20px; padding: 1rem 1.2rem; font-weight: 800; font-size: 1.15rem; margin-top: 0.5rem; }
    .result-approved { background: linear-gradient(135deg, #dcfce7, #ecfdf5); color: #166534; border: 1px solid #bbf7d0; }
    .result-rejected { background: linear-gradient(135deg, #fee2e2, #fff1f2); color: #991b1b; border: 1px solid #fecaca; }
    .good { color: #0f8a5f; font-weight: 700; }
    .bad { color: #c0392b; font-weight: 700; }
    .small-text { color: #5b6573; font-size: 0.92rem; }
    .applicant-header { color: #1d4ed8; font-size: 1.55rem; font-weight: 900; margin-bottom: 0.2rem; }
    .panel-header { color: #7c3aed; font-size: 1.55rem; font-weight: 900; margin-bottom: 0.2rem; }
    .stButton > button {
        width: 100%;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.8rem 1rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
        color: white !important;
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.28) !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 14px 30px rgba(37, 99, 235, 0.34) !important; }
    .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input { border-radius: 12px !important; border: 1px solid #d8e1ea !important; background-color: white !important; }
    .stExpander { border-radius: 16px; border: 1px solid #e1e8f0; background: rgba(255,255,255,0.85); }
    .stTab { padding-top: 1rem; }
    .stDataFrame { border-radius: 14px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    try:
        with open("best_model_name.pkl", "rb") as f:
            best_model_name = pickle.load(f)
    except FileNotFoundError:
        best_model_name = "Tuned LightGBM"
    return model, scaler, label_encoders, feature_columns, best_model_name

model, scaler, label_encoders, feature_columns, best_model_name = load_artifacts()

APPROVED_LABEL = 0
REJECTED_LABEL = 1

for k, v in {
    "prediction_done": False,
    "prediction": None,
    "approval_prob": None,
    "rejection_prob": None,
    "strengths": [],
    "risks": [],
    "suggestions": [],
    "summary_df": None,
    "user_inputs": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def generate_reason_signals(data):
    strengths, risks, suggestions = [], [], []
    if data["person_income"] >= 80000:
        strengths.append("Strong annual income compared with typical applicants")
    elif data["person_income"] < 50000:
        risks.append("Lower annual income may reduce repayment confidence")
        suggestions.append("Increase declared stable income or add verified co-applicant income")
    if data["credit_score"] >= 700:
        strengths.append("Good credit score supports approval")
    elif data["credit_score"] < 600:
        risks.append("Low credit score is a major risk signal")
        suggestions.append("Improve credit score by repaying dues on time and reducing outstanding balances")
    if data["loan_interest_rate"] <= 10:
        strengths.append("Lower interest rate profile is favorable")
    elif data["loan_interest_rate"] > 13:
        risks.append("Higher interest rate indicates a riskier borrowing profile")
        suggestions.append("Try applying for a lower-risk loan profile or improve creditworthiness before reapplying")
    if data["loan_percentage"] <= 0.15:
        strengths.append("Loan percentage is within a safer range")
    elif data["loan_percentage"] > 0.25:
        risks.append("High loan percentage suggests higher repayment burden")
        suggestions.append("Reduce the requested loan amount or increase income to improve the ratio")
    if data["credit_history"] >= 5:
        strengths.append("Adequate credit history supports decision confidence")
    elif data["credit_history"] < 3:
        risks.append("Limited credit history may weaken trust in repayment behavior")
        suggestions.append("Build stronger credit history with consistent repayment records")
    if data["employee_experience"] >= 3:
        strengths.append("Work experience suggests better income stability")
    elif data["employee_experience"] == 0:
        risks.append("No work experience may indicate lower repayment stability")
        suggestions.append("Show stable employment history or additional income proof")
    if data["previous_loan"] == "Yes":
        strengths.append("Previous loan history may support lending familiarity")
    return strengths, risks, list(dict.fromkeys(suggestions))

st.markdown("""
<div class="hero">
    <div class="hero-title">💳 Loan Approval Prediction</div>
    <div class="hero-subtitle">A cleaner dashboard for testing applicant details, predicting loan status, and reviewing decision signals.</div>
    <div class="pill-row">
        <span class="pill p1">Interactive Assessment</span>
        <span class="pill p2">Applicant Summary</span>
        <span class="pill p3">Decision Signals</span>
        <span class="pill p4">Improvement Tips</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.caption(f"Prediction app powered by: {best_model_name}")

top1, top2, top3, top4 = st.columns(4)
with top1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Target Task", "Classification")
    st.markdown('</div>', unsafe_allow_html=True)
with top2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Model Ready", "Yes")
    st.markdown('</div>', unsafe_allow_html=True)
with top3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Input Features", "13")
    st.markdown('</div>', unsafe_allow_html=True)
with top4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Output Labels", "Approved / Rejected")
    st.markdown('</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Prediction", "Applicant Summary", "Decision Signals"])

with tab1:
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown('<div class="card"><div class="applicant-header">Applicant Details</div>', unsafe_allow_html=True)
        with st.form("loan_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age", min_value=18, max_value=100, value=28)
                gender = st.selectbox("Gender", ["female", "male"])
                education = st.selectbox("Education", ["Associate", "Bachelor", "Doctorate", "High School", "Master"])
                person_income = st.number_input("Person Income", min_value=0.0, max_value=10000000.0, value=62000.0, step=1000.0)
                employee_experience = st.number_input("Employee Experience", min_value=0, max_value=60, value=4)
                home_onwership = st.selectbox("Home Onwership", ["MORTGAGE", "OTHER", "OWN", "RENT"])
            with c2:
                loan_amount = st.number_input("Loan Amount", min_value=0.0, max_value=35000.0, value=11000.0, step=500.0)
                loan_intent = st.selectbox("Loan Intent", ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])
                loan_interest_rate = st.number_input("Loan Interest Rate", min_value=0.0, max_value=20.0, value=10.8, step=0.1)
                loan_percentage = st.number_input("Loan Percentage", min_value=0.0, max_value=1.0, value=0.17, step=0.01)
                credit_history = st.number_input("Credit History", min_value=0, max_value=50, value=5)
                credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=668)
                previous_loan = st.selectbox("Previous Loan", ["No", "Yes"])
            submitted = st.form_submit_button("Predict Loan Status", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card"><div class="panel-header">Prediction Panel</div>', unsafe_allow_html=True)
        st.markdown("<div class='small-text'>Submit the form to generate a prediction, probability split, and decision summary.</div>", unsafe_allow_html=True)
        if st.session_state.prediction_done:
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            a1, a2, a3 = st.columns([1, 1, 1.6])
            with a1:
                st.metric("Approval Probability", f"{st.session_state.approval_prob:.2%}")
            with a2:
                st.metric("Rejection Probability", f"{st.session_state.rejection_prob:.2%}")
            with a3:
                st.metric("Model Used", best_model_name)
                st.markdown(f"<div class='small-text' style='margin-top:-0.35rem; white-space: normal; overflow: visible; text-overflow: initial; color:#7c3aed; font-weight:700;'>{best_model_name}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            if st.session_state.prediction == APPROVED_LABEL:
                st.markdown("<div class='result-approved'>✅ Loan Status Prediction: Approved</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-rejected'>❌ Loan Status Prediction: Rejected</div>", unsafe_allow_html=True)
        else:
            st.info("Run a prediction to see the result here.")
        st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        user_inputs = {
            "age": age, "gender": gender, "education": education, "person_income": person_income,
            "employee_experience": employee_experience, "home_onwership": home_onwership,
            "loan_amount": loan_amount, "loan_intent": loan_intent, "loan_interest_rate": loan_interest_rate,
            "loan_percentage": loan_percentage, "credit_history": credit_history,
            "credit_score": credit_score, "previous_loan": previous_loan
        }
        input_data = pd.DataFrame([{
            "age": age, "gender": gender, "education": education, "employee_experience": employee_experience,
            "home_onwership": home_onwership, "loan_amount": loan_amount, "loan_intent": loan_intent,
            "loan_interest_rate": loan_interest_rate, "loan_percentage": loan_percentage,
            "credit_history": credit_history, "credit_score": credit_score, "previous_loan": previous_loan,
            "person_income_log": np.log1p(person_income)
        }])
        categorical_cols = ["gender", "education", "home_onwership", "loan_intent", "previous_loan"]
        for col in categorical_cols:
            input_data[col] = label_encoders[col].transform(input_data[col])
        input_data = input_data[feature_columns]
        if "Logistic Regression" in best_model_name:
            input_processed = input_data.copy()
            numeric_features_only = input_processed.select_dtypes(include=np.number).columns.tolist()
            input_processed[numeric_features_only] = scaler.transform(input_processed[numeric_features_only])
        else:
            input_processed = input_data.copy()
        prediction = model.predict(input_processed)[0]
        probability = model.predict_proba(input_processed)[0]
        approval_prob = float(probability[APPROVED_LABEL])
        rejection_prob = float(probability[REJECTED_LABEL])
        strengths, risks, suggestions = generate_reason_signals(user_inputs)
        summary_df = pd.DataFrame({
            "Field": ["Age", "Gender", "Education", "Person Income", "Employee Experience", "Home Onwership", "Loan Amount", "Loan Intent", "Loan Interest Rate", "Loan Percentage", "Credit History", "Credit Score", "Previous Loan"],
            "Value": [age, gender, education, person_income, employee_experience, home_onwership, loan_amount, loan_intent, loan_interest_rate, loan_percentage, credit_history, credit_score, previous_loan]
        })
        st.session_state.prediction_done = True
        st.session_state.prediction = int(prediction)
        st.session_state.approval_prob = approval_prob
        st.session_state.rejection_prob = rejection_prob
        st.session_state.strengths = strengths
        st.session_state.risks = risks
        st.session_state.suggestions = suggestions
        st.session_state.summary_df = summary_df
        st.session_state.user_inputs = user_inputs
        st.rerun()

with tab2:
    st.markdown('<div class="card"><h3 class="section-title">Applicant Input Summary</h3>', unsafe_allow_html=True)
    if st.session_state.summary_df is not None:
        st.dataframe(st.session_state.summary_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run a prediction first to see the applicant summary.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card"><h3 class="section-title">Decision Signals</h3>', unsafe_allow_html=True)
    if st.session_state.prediction_done:
        st.write("These are supportive insights based on the entered values and project logic.")
    else:
        st.info("Run a prediction first to see decision signals.")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.prediction_done:
        if st.session_state.strengths:
            st.markdown("#### Positive Signals")
            for item in st.session_state.strengths:
                st.markdown(f"- <span class='good'>{item}</span>", unsafe_allow_html=True)
        if st.session_state.risks:
            st.markdown("#### Risk Signals")
            for item in st.session_state.risks:
                st.markdown(f"- <span class='bad'>{item}</span>", unsafe_allow_html=True)
        if not st.session_state.strengths and not st.session_state.risks:
            st.info("No strong signals detected from the simple rule summary.")
        if st.session_state.prediction == REJECTED_LABEL:
            st.markdown("#### Suggestions to Improve Approval Chances")
            if st.session_state.suggestions:
                for item in st.session_state.suggestions:
                    st.markdown(f"- {item}")
            else:
                st.info("Maintain better credit behavior, reduce repayment burden, and improve income stability before reapplying.")

st.caption("Note: Decision signals and suggestions are supportive insights based on entered values and project logic, not official bank lending advice.")
