import streamlit as st
import pandas as pd
from datetime import datetime
import os

# Configuration
CSV_FILE = "icu_patients.csv"

# Define all columns based on the medical dataset
COLUMNS = [
    "Timestamp", "Patient_ID", "Age", "Gender", "Insurance_Type", "Ethnicity", 
    "Marital_Status", "Hospital_Admission_Type", "First_Care_Unit", "ICD_Code_Category",
    "Heart_Rate", "Systolic_BP", "Diastolic_BP", "Respiratory_Rate", "Temperature",
    "SpO2", "Glucose", "White_Blood_Cells", "Hemoglobin", "Platelets",
    "Creatinine", "Sodium", "Potassium", "Bilirubin", "Lactate", "pH",
    "Notes"
]

# Dropdown options
GENDER_OPTIONS = ["GENDER_M", "GENDER_F"]
AGE_BINS = ["age_group_AGE_18-39", "age_group_AGE_40-59", "age_group_AGE_60-79", "age_group_AGE_80+"]
INSURANCE_OPTIONS = ["insurance_group_INS_Medicaid", "insurance_group_INS_Medicare", "insurance_group_INS_Other"]
ETHNICITY_OPTIONS = ["ethnicity_group_ETH_asian", "ethnicity_group_ETH_black", "ethnicity_group_ETH_latino", 
                     "ethnicity_group_ETH_other", "ethnicity_group_ETH_white"]
MARITAL_OPTIONS = ["marital_group_MAR_divorced", "marital_group_MAR_married", "marital_group_MAR_single",
                   "marital_group_MAR_unknown", "marital_group_MAR_widowed"]
ADMISSION_OPTIONS = ["admission_type_AMBULATORY OBSERVATION", "admission_type_DIRECT EMER.", 
                     "admission_type_DIRECT OBSERVATION", "admission_type_ELECTIVE", 
                     "admission_type_EU OBSERVATION", "admission_type_EW EMER.",
                     "admission_type_OBSERVATION ADMIT", "admission_type_SURGICAL SAME DAY ADMISSION",
                     "admission_type_URGENT"]
FIRST_CARE_OPTIONS = ["first_careunit_Cardiac Vascular Intensive Care Unit (CVICU)",
                      "first_careunit_Coronary Care Unit (CCU)",
                      "first_careunit_Medical Intensive Care Unit (MICU)",
                      "first_careunit_Medical/Surgical Intensive Care Unit (MICU/SICU)",
                      "first_careunit_Neuro Intermediate",
                      "first_careunit_Neuro Stepdown",
                      "first_careunit_Neuro Surgical Intensive Care Unit (Neuro SICU)",
                      "first_careunit_Surgical Intensive Care Unit (SICU)",
                      "first_careunit_Trauma SICU (TSICU)"]
ICD_CATEGORIES = ["Blood", "Circulatory", "Congenital", "Digestive", "Endocrine", "Genitourinary",
                  "Infectious", "Injury", "Mental", "Misc", "Muscular", "Neoplasms", "Nervous",
                  "Pregnancy", "Prenatal", "Respiratory", "Skin"]

# Initialize or migrate CSV file
def init_csv():
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            if list(df.columns) != COLUMNS:
                st.warning("Detected old CSV format. Creating backup and initializing new format...")
                backup_name = f"icu_patients_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                os.rename(CSV_FILE, backup_name)
                st.info(f"Old data backed up to: {backup_name}")
                df = pd.DataFrame(columns=COLUMNS)
                df.to_csv(CSV_FILE, index=False)
                st.success("New CSV file created with updated format!")
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            st.warning("Creating backup and starting fresh...")
            backup_name = f"icu_patients_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            os.rename(CSV_FILE, backup_name)
            df = pd.DataFrame(columns=COLUMNS)
            df.to_csv(CSV_FILE, index=False)
            st.success("New CSV file created!")
    else:
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(CSV_FILE, index=False)

# Load data with error handling
def load_data():
    try:
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
            if list(df.columns) == COLUMNS:
                return df
            else:
                st.error("CSV column mismatch detected!")
                return pd.DataFrame(columns=COLUMNS)
        return pd.DataFrame(columns=COLUMNS)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(columns=COLUMNS)

# Save data function
def save_data(data_dict):
    try:
        df = pd.DataFrame([data_dict])
        if os.path.exists(CSV_FILE):
            existing_df = pd.read_csv(CSV_FILE)
            if list(existing_df.columns) == COLUMNS:
                df.to_csv(CSV_FILE, mode="a", header=False, index=False)
            else:
                df.to_csv(CSV_FILE, mode="w", header=True, index=False)
        else:
            df.to_csv(CSV_FILE, mode="w", header=True, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")
        return False

# Main App
def main():
    st.set_page_config(page_title="ICU Patient Data Collection", layout="wide")
    st.title("ICU Patient Data Collection Dashboard")
    
    # Initialize session state for form reset
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False
    
    # Initialize CSV
    init_csv()
    
    # Sidebar
    with st.sidebar:
        st.header("Dashboard Controls")
        st.info("""
        **Instructions:**
        - Fill patient demographics and vital signs
        - Submit to save data locally
        - Download or preview collected data
        """)
        
        st.divider()
        
        # Statistics
        df = load_data()
        st.metric("Total Patients", len(df))
        
        if st.button("Refresh Data", use_container_width=True):
            st.rerun()
        
        # Option to reset CSV
        st.divider()
        if st.button("Reset Database", use_container_width=True, type="secondary"):
            if os.path.exists(CSV_FILE):
                backup_name = f"icu_patients_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                os.rename(CSV_FILE, backup_name)
                st.warning(f"Backed up to: {backup_name}")
            init_csv()
            st.success("Database reset!")
            st.rerun()
    
    # Show success message if form was just submitted
    if st.session_state.form_submitted:
        st.success("Patient data submitted successfully! Form ready for new patient.")
        st.session_state.form_submitted = False
    
    # Main form
    with st.form("patient_form", clear_on_submit=True):
        st.subheader("Patient Demographics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            patient_id = st.text_input("Patient ID*", placeholder="e.g., ICU-001")
            age_bin = st.selectbox("Age Group*", AGE_BINS)
            gender = st.selectbox("Gender*", GENDER_OPTIONS)
        
        with col2:
            insurance = st.selectbox("Insurance Type*", INSURANCE_OPTIONS)
            ethnicity = st.selectbox("Ethnicity*", ETHNICITY_OPTIONS)
            marital = st.selectbox("Marital Status*", MARITAL_OPTIONS)
        
        with col3:
            admission_type = st.selectbox("Admission Type*", ADMISSION_OPTIONS)
            first_care = st.selectbox("First Care Unit*", FIRST_CARE_OPTIONS)
            icd_category = st.selectbox("ICD Code Category*", ICD_CATEGORIES)
        
        st.divider()
        st.subheader("Vital Signs & Lab Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            heart_rate = st.number_input("Heart Rate (bpm)", 0, 300, 80)
            systolic_bp = st.number_input("Systolic BP (mmHg)", 0, 300, 120)
            respiratory_rate = st.number_input("Respiratory Rate", 0, 100, 16)
            spo2 = st.number_input("SpO2 (%)", 0.0, 100.0, 98.0, 0.1)
            white_blood_cells = st.number_input("WBC (K/µL)", 0.0, 50.0, 8.0, 0.1)
            creatinine = st.number_input("Creatinine (mg/dL)", 0.0, 20.0, 1.0, 0.1)
        
        with col2:
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", 0, 200, 80)
            temperature = st.number_input("Temperature (°C)", 30.0, 45.0, 37.0, 0.1)
            glucose = st.number_input("Glucose (mg/dL)", 0, 1000, 100)
            hemoglobin = st.number_input("Hemoglobin (g/dL)", 0.0, 25.0, 14.0, 0.1)
            sodium = st.number_input("Sodium (mEq/L)", 0, 200, 140)
        
        with col3:
            platelets = st.number_input("Platelets (K/µL)", 0, 1000, 250)
            potassium = st.number_input("Potassium (mEq/L)", 0.0, 10.0, 4.0, 0.1)
            bilirubin = st.number_input("Bilirubin (mg/dL)", 0.0, 30.0, 1.0, 0.1)
        
        with col4:
            lactate = st.number_input("Lactate (mmol/L)", 0.0, 20.0, 1.5, 0.1)
            ph = st.number_input("pH", 6.0, 8.0, 7.4, 0.01)
        
        st.divider()
        notes = st.text_area("Additional Notes", height=100, placeholder="Any additional observations or comments...")
        
        submitted = st.form_submit_button("Submit Patient Data", use_container_width=True, type="primary")
    
    # Handle form submission
    if submitted:
        if not patient_id:
            st.error("Patient ID is required!")
        else:
            data_dict = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Patient_ID": patient_id,
                "Age": age_bin,
                "Gender": gender,
                "Insurance_Type": insurance,
                "Ethnicity": ethnicity,
                "Marital_Status": marital,
                "Hospital_Admission_Type": admission_type,
                "First_Care_Unit": first_care,
                "ICD_Code_Category": icd_category,
                "Heart_Rate": heart_rate,
                "Systolic_BP": systolic_bp,
                "Diastolic_BP": diastolic_bp,
                "Respiratory_Rate": respiratory_rate,
                "Temperature": temperature,
                "SpO2": spo2,
                "Glucose": glucose,
                "White_Blood_Cells": white_blood_cells,
                "Hemoglobin": hemoglobin,
                "Platelets": platelets,
                "Creatinine": creatinine,
                "Sodium": sodium,
                "Potassium": potassium,
                "Bilirubin": bilirubin,
                "Lactate": lactate,
                "pH": ph,
                "Notes": notes
            }
            
            if save_data(data_dict):
                st.session_state.form_submitted = True
                st.rerun()
            else:
                st.error("Failed to save data. Please try again.")
    
    # Data management section
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download CSV", use_container_width=True):
            df = load_data()
            if len(df) > 0:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download icu_patients.csv",
                    data=csv,
                    file_name=f"icu_patients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("No data to download!")
    
    with col2:
        show_preview = st.checkbox("Preview Collected Data")
    
    if show_preview:
        df = load_data()
        if len(df) == 0:
            st.info("No patient data yet. Submit entries to see them here!")
        else:
            st.dataframe(df, use_container_width=True, height=400)
            st.caption(f"Total records: {len(df)} | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()