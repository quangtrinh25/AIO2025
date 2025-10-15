import streamlit as st
import pandas as pd
from datetime import datetime
import os
import io

# Configuration
CSV_FILE = "icu_patients.csv"
TEMP_CSV_FILE = "temp_prediction_data.csv"

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

# Save temporary CSV for prediction
def save_temp_csv(df):
    try:
        df.to_csv(TEMP_CSV_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving temporary CSV: {str(e)}")
        return False

# Load temporary CSV
def load_temp_csv():
    try:
        if os.path.exists(TEMP_CSV_FILE):
            return pd.read_csv(TEMP_CSV_FILE)
        return None
    except Exception as e:
        st.error(f"Error loading temporary CSV: {str(e)}")
        return None

# Get unique patient IDs
def get_patient_ids():
    df = load_data()
    if len(df) > 0:
        return sorted(df['Patient_ID'].unique().tolist())
    return []

# Get latest patient data
def get_patient_data(patient_id):
    df = load_data()
    patient_records = df[df['Patient_ID'] == patient_id]
    if len(patient_records) > 0:
        return patient_records.iloc[-1].to_dict()
    return None

# Save data function (for new patient)
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

# Update patient data
def update_patient_data(patient_id, data_dict):
    try:
        df = load_data()
        df = df[df['Patient_ID'] != patient_id]
        new_df = pd.DataFrame([data_dict])
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Error updating data: {str(e)}")
        return False

# Delete patient record
def delete_patient_record(patient_id, timestamp):
    try:
        df = load_data()
        df = df[~((df['Patient_ID'] == patient_id) & (df['Timestamp'] == timestamp))]
        df.to_csv(CSV_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Error deleting record: {str(e)}")
        return False

# CSV Editor Tab
def csv_editor_tab():
    st.header("CSV Data Editor")
    
    df = load_data()
    
    if len(df) == 0:
        st.warning("No data available to edit. Please add some patient records first.")
        return
    
    # Display current data with data editor
    st.subheader("Edit Data Directly")
    st.info("Changes are made in memory. Click 'Save Changes to Main CSV' to persist changes.")
    
    # Create editable dataframe with string type for Timestamp
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        height=400,
        column_config={
            "Timestamp": st.column_config.TextColumn("Timestamp"),
            "Age": st.column_config.NumberColumn("Age", min_value=0, max_value=120),
            "Heart_Rate": st.column_config.NumberColumn("Heart Rate", min_value=0, max_value=300),
            "Systolic_BP": st.column_config.NumberColumn("Systolic BP", min_value=0, max_value=300),
            "Diastolic_BP": st.column_config.NumberColumn("Diastolic BP", min_value=0, max_value=200),
            "Temperature": st.column_config.NumberColumn("Temperature", min_value=30.0, max_value=45.0, format="%.1f"),
            "SpO2": st.column_config.NumberColumn("SpO2", min_value=0.0, max_value=100.0, format="%.1f"),
        }
    )
    
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Save Changes to Main CSV", use_container_width=True, type="primary"):
            try:
                edited_df.to_csv(CSV_FILE, index=False)
                st.success("Changes saved to main CSV successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error saving changes: {str(e)}")
    
    with col2:
        if st.button("Save as Temp CSV for Prediction", use_container_width=True):
            if save_temp_csv(edited_df):
                st.success(f"Temporary CSV saved: {TEMP_CSV_FILE}")
                st.info("This file can now be used for model prediction.")
            else:
                st.error("Failed to save temporary CSV")
    
    with col3:
        if st.button("Reset to Original", use_container_width=True):
            st.info("Resetting to original data...")
            st.rerun()
    
    with col4:
        csv_data = edited_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Edited CSV",
            data=csv_data,
            file_name=f"edited_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Show temp CSV info
    st.divider()
    temp_df = load_temp_csv()
    if temp_df is not None:
        with st.expander("Current Temporary Prediction CSV"):
            st.dataframe(temp_df, use_container_width=True)
            st.caption(f"Records: {len(temp_df)} | File: {TEMP_CSV_FILE}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Delete Temp CSV", use_container_width=True):
                    try:
                        os.remove(TEMP_CSV_FILE)
                        st.success("Temporary CSV deleted!")
                        st.rerun()
                    except:
                        st.error("Failed to delete temporary CSV")
            
            with col2:
                temp_csv_data = temp_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Temp CSV",
                    data=temp_csv_data,
                    file_name=TEMP_CSV_FILE,
                    mime="text/csv",
                    use_container_width=True
                )

# Main App
def main():
    st.set_page_config(page_title="ICU Patient Data Collection", layout="wide")
    st.title("ICU Patient Data Collection Dashboard")
    
    # Initialize session state
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False
    if 'mode' not in st.session_state:
        st.session_state.mode = 'new'
    
    # Initialize CSV
    init_csv()
    
    # Sidebar
    with st.sidebar:
        st.header("Dashboard Controls")
        
        # Tab selector
        tab_mode = st.radio(
            "Select Mode:",
            ["Add New Patient", "Update Patient", "CSV Editor", "View & Manage"],
            index=0
        )
        
        st.divider()
        
        # Statistics
        df = load_data()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Patients", len(df['Patient_ID'].unique()) if len(df) > 0 else 0)
        with col2:
            st.metric("Total Records", len(df))
        
        # Check for temp CSV
        if os.path.exists(TEMP_CSV_FILE):
            temp_df = load_temp_csv()
            if temp_df is not None:
                st.success(f"Temp CSV: {len(temp_df)} records")
        
        st.divider()
        
        if st.button("Refresh Data", use_container_width=True):
            st.rerun()
        
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
        st.success("Operation completed successfully!")
        st.session_state.form_submitted = False
    
    # Route to different tabs
    if tab_mode == "CSV Editor":
        csv_editor_tab()
    
    elif tab_mode == "View & Manage":
        st.header("View & Manage Data")
        
        df = load_data()
        if len(df) == 0:
            st.info("No patient data yet. Submit entries to see them here!")
        else:
            # Search and filter
            col1, col2, col3 = st.columns(3)
            with col1:
                search_id = st.text_input("Search by Patient ID")
            with col2:
                filter_unit = st.multiselect("Filter by Care Unit", 
                                            options=df['First_Care_Unit'].unique().tolist() if len(df) > 0 else [])
            with col3:
                filter_category = st.multiselect("Filter by ICD Category",
                                                options=df['ICD_Code_Category'].unique().tolist() if len(df) > 0 else [])
            
            # Apply filters
            filtered_df = df.copy()
            if search_id:
                filtered_df = filtered_df[filtered_df['Patient_ID'].str.contains(search_id, case=False, na=False)]
            if filter_unit:
                filtered_df = filtered_df[filtered_df['First_Care_Unit'].isin(filter_unit)]
            if filter_category:
                filtered_df = filtered_df[filtered_df['ICD_Code_Category'].isin(filter_category)]
            
            st.dataframe(filtered_df, use_container_width=True, height=400)
            st.caption(f"Showing {len(filtered_df)} of {len(df)} records")
            
            # Bulk actions
            st.divider()
            st.subheader("Delete Records")
            
            if len(filtered_df) > 0:
                col1, col2 = st.columns([3, 1])
                with col1:
                    selected_records = st.multiselect(
                        "Select records to delete (Patient_ID - Timestamp):",
                        options=[f"{row['Patient_ID']} - {row['Timestamp']}" for _, row in filtered_df.iterrows()]
                    )
                with col2:
                    st.write("")
                    st.write("")
                    if st.button("Delete Selected", type="secondary", use_container_width=True):
                        if selected_records:
                            success_count = 0
                            for record in selected_records:
                                parts = record.split(" - ", 1)  # Split only on first occurrence
                                if len(parts) == 2:
                                    patient_id, timestamp = parts
                                    if delete_patient_record(patient_id, timestamp):
                                        success_count += 1
                            if success_count > 0:
                                st.success(f"Deleted {success_count} record(s)!")
                                st.rerun()
                            else:
                                st.error("Failed to delete records!")
                        else:
                            st.warning("Please select records to delete")
            
            # Download options
            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Filtered Data",
                    data=csv,
                    file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col2:
                csv_all = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download All Data",
                    data=csv_all,
                    file_name=f"all_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col3:
                if st.button("Save All as Temp CSV", use_container_width=True):
                    if save_temp_csv(df):
                        st.success("Saved as temporary prediction CSV!")
    
    elif tab_mode == "Update Patient":
        st.header("Update Existing Patient")
        patient_ids = get_patient_ids()
        
        if len(patient_ids) == 0:
            st.warning("No patients in database. Please add a new patient first.")
        else:
            selected_patient = st.selectbox("Select Patient ID to Update:", patient_ids)
            
            if selected_patient:
                patient_data = get_patient_data(selected_patient)
                
                with st.expander("View Patient Current Data"):
                    df = load_data()
                    patient_history = df[df['Patient_ID'] == selected_patient]
                    st.dataframe(patient_history, use_container_width=True)
                
                with st.form("update_form", clear_on_submit=True):
                    st.subheader(f"Update Patient: {selected_patient}")
                    st.info(f"Last updated: {patient_data['Timestamp']}")
                    
                    st.divider()
                    st.subheader("Vital Signs & Lab Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        heart_rate = st.number_input("Heart Rate (bpm)", 0, 300, int(patient_data['Heart_Rate']))
                        systolic_bp = st.number_input("Systolic BP (mmHg)", 0, 300, int(patient_data['Systolic_BP']))
                        respiratory_rate = st.number_input("Respiratory Rate", 0, 100, int(patient_data['Respiratory_Rate']))
                        spo2 = st.number_input("SpO2 (%)", 0.0, 100.0, float(patient_data['SpO2']), 0.1)
                        white_blood_cells = st.number_input("WBC (K/µL)", 0.0, 50.0, float(patient_data['White_Blood_Cells']), 0.1)
                        creatinine = st.number_input("Creatinine (mg/dL)", 0.0, 20.0, float(patient_data['Creatinine']), 0.1)
                    
                    with col2:
                        diastolic_bp = st.number_input("Diastolic BP (mmHg)", 0, 200, int(patient_data['Diastolic_BP']))
                        temperature = st.number_input("Temperature (°C)", 30.0, 45.0, float(patient_data['Temperature']), 0.1)
                        glucose = st.number_input("Glucose (mg/dL)", 0, 1000, int(patient_data['Glucose']))
                        hemoglobin = st.number_input("Hemoglobin (g/dL)", 0.0, 25.0, float(patient_data['Hemoglobin']), 0.1)
                        sodium = st.number_input("Sodium (mEq/L)", 0, 200, int(patient_data['Sodium']))
                    
                    with col3:
                        platelets = st.number_input("Platelets (K/µL)", 0, 1000, int(patient_data['Platelets']))
                        potassium = st.number_input("Potassium (mEq/L)", 0.0, 10.0, float(patient_data['Potassium']), 0.1)
                        bilirubin = st.number_input("Bilirubin (mg/dL)", 0.0, 30.0, float(patient_data['Bilirubin']), 0.1)
                    
                    with col4:
                        lactate = st.number_input("Lactate (mmol/L)", 0.0, 20.0, float(patient_data['Lactate']), 0.1)
                        ph = st.number_input("pH", 6.0, 8.0, float(patient_data['pH']), 0.01)
                    
                    st.divider()
                    notes = st.text_area("Additional Notes", height=100, value=str(patient_data.get('Notes', '')))
                    
                    update_button = st.form_submit_button("Update Patient Data", use_container_width=True, type="primary")
                
                if update_button:
                    updated_data = {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Patient_ID": selected_patient,
                        "Age": patient_data['Age'],
                        "Gender": patient_data['Gender'],
                        "Insurance_Type": patient_data['Insurance_Type'],
                        "Ethnicity": patient_data['Ethnicity'],
                        "Marital_Status": patient_data['Marital_Status'],
                        "Hospital_Admission_Type": patient_data['Hospital_Admission_Type'],
                        "First_Care_Unit": patient_data['First_Care_Unit'],
                        "ICD_Code_Category": patient_data['ICD_Code_Category'],
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
                    
                    if update_patient_data(selected_patient, updated_data):
                        st.session_state.form_submitted = True
                        st.rerun()
    
    else:  # Add New Patient
        st.header("Add New Patient")
        
        with st.form("patient_form", clear_on_submit=True):
            st.subheader("Patient Demographics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                patient_id = st.text_input("Patient ID*", placeholder="e.g., ICU-001")
                age = st.number_input("Age (years)*", min_value=60, max_value=120, value=65, step=1)
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
        
        if submitted:
            if not patient_id:
                st.error("Patient ID is required!")
            else:
                existing_ids = get_patient_ids()
                if patient_id in existing_ids:
                    st.error(f"Patient ID '{patient_id}' already exists! Please use 'Update Patient' mode or choose a different ID.")
                else:
                    data_dict = {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Patient_ID": patient_id,
                        "Age": age,
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

if __name__ == "__main__":
    main()
