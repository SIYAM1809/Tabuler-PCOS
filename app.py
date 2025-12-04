import streamlit as st
import pandas as pd
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="PCOS Detection System", page_icon="🩺", layout="wide")

# --- 1. LOAD THE TRAINED PIPELINE ---
@st.cache_resource
def load_model():
    try:
        # Load the pipeline you saved from Colab
        return joblib.load('pcos_pipelineV3.pkl')
    except FileNotFoundError:
        st.error("⚠️ Model file 'pcos_pipelineV3.pkl' not found. Please place it in the same folder as this script.")
        return None

model = load_model()

# --- 2. DEFINE COLUMNS ---
# Exact columns from your training data
ALL_COLUMNS = [
    'Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'BMI', 'Blood Group', 'Pulse rate(bpm)', 
    'RR (breaths/min)', 'Hb(g/dl)', 'Cycle(R/I)', 'Cycle length(days)', 'Marraige Status (Yrs)', 
    'Pregnant(Y/N)', 'No. of abortions', 'I   beta-HCG(mIU/mL)', 'II    beta-HCG(mIU/mL)', 
    'FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH', 'Hip(inch)', 'Waist(inch)', 'Waist:Hip Ratio', 
    'TSH (mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)', 'Vit D3 (ng/mL)', 'PRG(ng/mL)', 'RBS(mg/dl)', 
    'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)', 
    'Pimples(Y/N)', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)', 'BP _Systolic (mmHg)', 
    'BP _Diastolic (mmHg)', 'Follicle No. (L)', 'Follicle No. (R)', 'Avg. F size (L) (mm)', 
    'Avg. F size (R) (mm)', 'Endometrium (mm)'
]

# --- 3. UI HELPER FUNCTIONS ---

def get_user_input():
    st.write("### 📋 Patient Details")
    
    # We use a form to collect all data before predicting
    with st.form("pcos_form"):
        
        # Split form into 3 columns for better layout
        c1, c2, c3 = st.columns(3)
        cols_iter = [c1, c2, c3]
        
        user_data = {}
        
        for i, col_name in enumerate(ALL_COLUMNS):
            col_container = cols_iter[i % 3] # Rotate through columns
            
            # --- CUSTOM LOGIC FOR SPECIFIC COLUMNS ---
            
            # 1. Handle Binary Yes/No Columns
            if "(Y/N)" in col_name:
                val = col_container.radio(
                    label=col_name,
                    options=["No", "Yes"],
                    horizontal=True,
                    key=col_name
                )
                # Convert to 1 (Yes) or 0 (No)
                user_data[col_name] = 1 if val == "Yes" else 0
                
            # 2. Handle Cycle (Regular/Irregular) - Standard dataset uses 2 and 4
            elif "Cycle(R/I)" in col_name:
                val = col_container.radio(
                    label=col_name,
                    options=["Regular", "Irregular"],
                    horizontal=True,
                    key=col_name
                )
                # Map to standard dataset values: Regular=2, Irregular=4
                user_data[col_name] = 2 if val == "Regular" else 4
                
            # 3. Handle the 'Dirty' Categorical Column (II beta-HCG)
            elif "II    beta-HCG" in col_name:
                # This column often has typos in the original dataset, treating it as text
                user_data[col_name] = col_container.text_input(col_name, value="1.99")
            
            # 4. Standard Numeric Inputs
            else:
                user_data[col_name] = col_container.number_input(
                    label=col_name, 
                    value=0.0, 
                    step=0.1
                )
        
        # Submit Button
        submit = st.form_submit_button("🔍 Analyze Risk")
        
    return user_data, submit

# --- 4. MAIN APP LOGIC ---

st.title("🧬 PCOS Diagnostic Tool")
st.markdown("Enter patient metrics below to predict Polycystic Ovary Syndrome risk.")

if model:
    # Get Input
    input_dict, pressed = get_user_input()
    
    if pressed:
        # Convert dictionary to DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Ensure column order matches training exactly
        input_df = input_df[ALL_COLUMNS]

        st.divider()
        st.subheader("Results")
        
        try:
            with st.spinner("Analyzing parameters..."):
                # Prediction
                prediction = model.predict(input_df)[0]
                
                # Probability (if supported by your model)
                try:
                    probability = model.predict_proba(input_df)[0][1]
                    confidence_score = f"{probability*100:.1f}%"
                except:
                    confidence_score = "N/A"

            # Display Output
            if prediction == 1: # Assuming 1 = PCOS/High Risk (Check your specific mapping!)
                st.error(f"🚨 **High Risk Detected**")
                st.write(f"The model suggests a high probability ({confidence_score}) of PCOS.")
                st.warning("Recommendation: Please consult a gynecologist for clinical correlation.")
            else:
                st.success(f"✅ **Low Risk Detected**")
                st.write(f"The model suggests the patient is likely Healthy ({confidence_score}).")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.info("Tip: Check 'II beta-HCG' input. Ensure it matches the format in your training data.")