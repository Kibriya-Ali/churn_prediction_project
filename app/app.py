import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
# 1. Page Configuration
st.set_page_config(page_title="KKBox Churn Predictor", layout="centered")
st.title("🎧 Customer Churn Prediction Engine")
st.markdown("Predict the likelihood of a user canceling their music streaming subscription.")

# 2. Load the Model
@st.cache_resource
def load_model():
    # Adjust path if running from a different directory
    return joblib.load(models/lgbm_churn_model.pkl)

model = load_model()

# 3. User Input Sidebar (Simulating a Customer Profile)
st.sidebar.header("Customer Profile")

# Create interactive sliders and dropdowns for key features
auto_renew = st.sidebar.selectbox("Auto-Renew Enabled?", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", ["Credit Card", "Convenience Store", "Telecom", "Other"])
total_secs = st.sidebar.slider("Total Listening Minutes (Last 30 Days)", 0, 5000, 1200)
completion_ratio = st.sidebar.slider("Song Completion Ratio", 0.0, 1.0, 0.75)
songs_skipped = st.sidebar.slider("Total Songs Skipped", 0, 500, 45)

# 4. Data Preprocessing for the Model
auto_renew_val = 1 if auto_renew == "Yes" else 0
payment_map = {"Credit Card": 41, "Convenience Store": 35, "Telecom": 39, "Other": 0}

# Create a dictionary with baseline values for all features the model expects
input_dict = {col: 0 for col in model.feature_name_}

# Set the fallback string for demographic categories we aren't asking the user for
for col in ['city', 'gender', 'registered_via']:
    if col in input_dict:
        input_dict[col] = 'Unknown'

# Overwrite the baseline with the user's actual slider/dropdown inputs
input_dict['is_auto_renew'] = auto_renew_val
input_dict['payment_method_id'] = payment_map[payment_method]
input_dict['total_secs'] = total_secs * 60
input_dict['completion_ratio'] = completion_ratio
input_dict['total_songs_skipped'] = songs_skipped
input_dict['historical_tx_count'] = 5
input_dict['total_songs_finished'] = int(songs_skipped * (completion_ratio / (1 - completion_ratio + 1e-5)))

# Convert to DataFrame ensuring the exact column order
input_data = pd.DataFrame([input_dict], columns=model.feature_name_)

# THE FIX: Apply the EXACT category metadata from the trained model
cat_idx = 0
for col in model.feature_name_:
    if col in ['city', 'gender', 'registered_via', 'payment_method_id']:
        # LightGBM saves the exact training categories in booster_.pandas_categorical
        # This forces our 1-row DataFrame to perfectly match the training schema
        input_data[col] = pd.Categorical(
            input_data[col], 
            categories=model.booster_.pandas_categorical[cat_idx]
        )
        cat_idx += 1

# 5. Prediction and Business Logic
if st.button("Predict Churn Risk"):
    churn_prob = model.predict_proba(input_data)[0][1]
    
    st.subheader("Prediction Results")
    if churn_prob > 0.6:
        st.error(f"High Risk of Churn: {churn_prob:.1%}")
        st.markdown("**Business Action:** Immediate intervention required. Trigger an automated email offering a 20% discount if they switch to a credit card.")
    elif churn_prob > 0.3:
        st.warning(f"Moderate Risk: {churn_prob:.1%}")
        st.markdown("**Business Action:** User is losing interest. Send a personalized 'Discover Weekly' playlist push notification to boost engagement.")
    else:
        st.success(f"Low Risk (Safe): {churn_prob:.1%}")
        st.markdown("**Business Action:** Do not offer discounts. Customer is highly engaged.")

    # --- NEW: Live SHAP Waterfall Plot ---
    st.markdown("---")
    st.subheader("Why did the model make this prediction?")
    st.markdown("This waterfall plot shows how this specific user's metrics pushed their churn risk up (red) or down (blue) from the baseline average.")
    
    with st.spinner("Calculating feature impacts..."):
        # Initialize the TreeExplainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values for our 1-row input_data dataframe
        shap_values = explainer(input_data)
        
        # LightGBM binary classification sometimes returns a 3D array [samples, features, classes]
        # We need to extract the positive class (Churn = 1) for the first (and only) row
        if len(shap_values.shape) > 2:
            shap_explanation = shap_values[:, :, 1][0]
        else:
            shap_explanation = shap_values[0]
            
        # Generate the plot
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(shap_explanation, max_display=10, show=False)
        
        # Render the matplotlib figure in Streamlit
        st.pyplot(fig)
        
        # Clear the figure from memory to prevent memory leaks during multiple clicks
        plt.clf()
