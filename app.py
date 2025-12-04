import streamlit as st
import pandas as pd
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PCOS Detection System", 
    page_icon="🩺", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. LOAD THE TRAINED PIPELINE ---
@st.cache_resource
def load_model():
    try:
        # Load the pipeline
        return joblib.load('pcos_pipeline_v3.pkl')
    except FileNotFoundError:
        st.error("⚠️ Model file 'pcos_pipeline_v3.pkl' not found. Please place it in the same folder as this script.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

model = load_model()

# --- 2. DEFINE COLUMNS (EXACT MATCH TO MODEL) ---
# These names have been verified against your .pkl file
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
    st.markdown("### 📋 Enter Patient Vitals & History")
    
    with st.form("pcos_form"):
        # Split form into 3 columns for a clean layout
        c1, c2, c3 = st.columns(3)
        cols_iter = [c1, c2, c3]
        
        user_data = {}
        
        for i, col_name in enumerate(ALL_COLUMNS):
            col_container = cols_iter[i % 3] # Rotate columns
            
            # --- LOGIC FOR SPECIFIC COLUMNS ---
            
            # 1. Binary Yes/No Columns
            if "(Y/N)" in col_name:
                val = col_container.radio(
                    label=col_name,
                    options=["No", "Yes"],
                    horizontal=True,
                    key=col_name
                )
                user_data[col_name] = 1 if val == "Yes" else 0
                
            # 2. Cycle (Regular/Irregular)
            elif "Cycle(R/I)" in col_name:
                val = col_container.radio(
                    label=col_name,
                    options=["Regular", "Irregular"],
                    horizontal=True,
                    key=col_name
                )
                # Standard dataset mapping: Regular=2, Irregular=4
                user_data[col_name] = 2 if val == "Regular" else 4
                
            # 3. Handling the Spacing Issues for Beta-HCG
            # Note: We use 'in' to catch it regardless of exact whitespace, 
            # but the key in user_data matches ALL_COLUMNS exactly.
            elif "beta-HCG" in col_name:
                user_data[col_name] = col_container.number_input(
                    label=col_name, 
                    value=1.99, 
                    step=0.01,
                    format="%.2f"
                )
            
            # 4. Standard Numeric Inputs
            else:
                user_data[col_name] = col_container.number_input(
                    label=col_name, 
                    value=0.0, 
                    step=0.1
                )
        
        st.markdown("---")
        submit = st.form_submit_button("🔍 Analyze Risk Factor", type="primary")
        
    return user_data, submit

# --- 4. MAIN APP LOGIC ---

# Sidebar Info
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
    st.title("PCOS Detector")
    st.info(
        """
        This AI tool assists in the early detection of Polycystic Ovary Syndrome (PCOS) 
        using clinical parameters.
        """
    )
    st.write("---")
    if model:
        st.success("✅ Model Loaded Successfully")
    else:
        st.error("❌ Model Missing")

st.title("🧬 AI-Powered PCOS Diagnostic Tool")

if model:
    # Get Input
    input_dict, pressed = get_user_input()
    
    if pressed:
        # Convert dictionary to DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # DOUBLE CHECK: Ensure columns are in the exact order the model expects
        input_df = input_df[ALL_COLUMNS]

        st.divider()
        st.subheader("Diagnostic Results")
        
        col_res1, col_res2 = st.columns([1, 2])
        
        try:
            with st.spinner("Processing clinical parameters..."):
                # Prediction
                prediction = model.predict(input_df)[0]
                
                # Probability (try/except in case model doesn't support probability)
                try:
                    probs = model.predict_proba(input_df)[0]
                    # Assuming class 1 is PCOS
                    probability = probs[1]
                    confidence_score = f"{probability*100:.1f}%"
                except:
                    probability = None
                    confidence_score = "N/A"

            # Display Output
            with col_res1:
                if prediction == 1: 
                    st.error("🚨 **RESULT: POSITIVE**")
                    st.metric(label="Risk Probability", value=confidence_score)
                else:
                    st.success("✅ **RESULT: NEGATIVE**")
                    st.metric(label="Risk Probability", value=confidence_score)

            with col_res2:
                if prediction == 1:
                    st.warning(
                        """
                        **The model predicts a high likelihood of PCOS.**
                        
                        **Recommended Next Steps:**
                        * Consult a Gynecologist for a pelvic ultrasound.
                        * Review hormonal profile (FSH, LH, Testosterone).
                        * Monitor lifestyle factors (Diet & Exercise).
                        """
                    )
                else:
                    st.success(
                        """
                        **The model predicts a low likelihood of PCOS.**
                        
                        **Note:** While the risk is low, please maintain a healthy lifestyle. 
                        If symptoms persist, consult a doctor.
                        """
                    )
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.write("Debug info:", e)

else:
    st.warning("Please upload the 'pcos_pipeline_v3.pkl' file to the root directory to continue.")