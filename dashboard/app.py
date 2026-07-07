# dashboard/app.py
"""
Interactive UI panel exposing user form layouts 
and connecting directly to active FastAPI endpoints.
"""

import requests
import streamlit as st

st.set_page_config(
    page_title="Institutional Credit Underwriting Desk",
    layout="wide"
)

st.title("🏦 Institutional Credit Underwriting Desk")
st.markdown("---")

# Pointing to standard docker runtime localhost network port settings
TARGET_API_ENDPOINT = "http://localhost:8000/api/v1/predict"

left_panel, right_panel = st.columns(2, gap="large")

with left_panel:
    st.subheader("📝 Applicant Demographics & Profile Input")
    
    app_id = st.number_input("Application Record ID (Id)", min_value=1, value=1001, step=1)
    gross_income = st.number_input("Annual Gross Income (Income)", min_value=0, value=60000, step=2500)
    applicant_age = st.slider("Chronological Age (Age)", 18, 100, 35)
    total_exp = st.slider("Total Career Experience (Experience - Years)", 0, 50, 10)
    
    marital_status = st.selectbox("Marital Configuration Status", ["single", "married"])
    home_status = st.selectbox("Residential Ownership Type", ["rented", "owned", "norent_noown"])
    fleet_status = st.selectbox("Automotive Asset Fleet Status", ["no", "yes"])
    
    profession_title = st.text_input("Stated Employment Profession Title", "Engineer")
    target_city = st.text_input("Target Municipal City Identifier", "Austin")
    target_state = st.text_input("Zoning State Region Code", "Texas")
    
    tenure_job = st.slider("Current Employment Tenure (Years)", 0, 30, 4)
    tenure_house = st.slider("Current House Residency Duration (Years)", 0, 30, 6)

with right_panel:
    st.subheader("📊 Automated Operational Underwriting Assessment")
    st.markdown("Click below to process real-time input fields through the model pipelines.")
    
    if st.button("Execute Credit Risk Assessment Pipeline", use_container_width=True):
        # Format payload dictionary keys to match Pydantic strict structures exactly
        request_payload = {
            "Id": app_id,
            "Income": float(gross_income),
            "Age": int(applicant_age),
            "Experience": int(total_exp),
            "Married/Single": str(marital_status),
            "House_Ownership": str(home_status),
            "Car_Ownership": str(fleet_status),
            "Profession": str(profession_title),
            "CITY": str(target_city),
            "STATE": str(target_state),
            "CURRENT_JOB_YRS": int(tenure_job),
            "CURRENT_HOUSE_YRS": int(tenure_house)
        }
        
        with st.spinner("Streaming data to prediction engine..."):
            try:
                network_response = requests.post(TARGET_API_ENDPOINT, json=request_payload, timeout=12)
                
                if network_response.status_code == 200:
                    payload_response = network_response.json()
                    underwriting_decision = payload_response["underwriting_decision"]
                    approval_probability = payload_response["approval_probability"]
                    risk_drivers = payload_response["primary_risk_drivers"]
                    
                    if "Approved" in underwriting_decision:
                        st.success(f"**Decision Summary**: {underwriting_decision}")
                    else:
                        st.error(f"**Decision Summary**: {underwriting_decision}")
                    
                    st.metric(
                        label="Calculated Baseline Approval Probability",
                        value=f"{approval_probability * 100:.2f}%",
                        delta=f"Risk: {(1.0 - approval_probability) * 100:.1f}%",
                        delta_color="inverse"
                    )
                    
                    st.markdown("### 🔍 Model Explanation Attributions")
                    st.write("Top 3 local preprocessed features driving this classification score:")
                    for rank, trait in enumerate(risk_drivers, start=1):
                        st.markdown(f"**{rank}.** `{trait}`")
                else:
                    st.error(f"Endpoint Error Code: {network_response.status_code}")
                    st.code(network_response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Connection Refused: Verify that the FastAPI backend application is active on port 8000.")