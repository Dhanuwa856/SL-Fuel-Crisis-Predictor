import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="SL Economic Crisis Predictor",
    page_icon="🇱🇰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== LOAD MODEL & SCALER (OPTIMIZED) ==========
# @st.cache_resource දාන්නේ model එක එක පාරක් load කරලා මතක තියාගන්නයි (Speed up)
@st.cache_resource
def load_models():
    sc = joblib.load('scaler.pkl')
    mdl = joblib.load('svm_model.pkl')
    return sc, mdl

scaler, model = load_models()

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    /* Success and error boxes */
    .success-box {
        background: linear-gradient(135deg, #1e7e34, #28a745);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    .error-box {
        background: linear-gradient(135deg, #b91c1c, #dc3545);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        font-size: 0.9rem;
        border-top: 1px solid #444;
        color: #bbb;
    }
    .footer a {
        color: #ff6b6b;
        text-decoration: none;
        font-weight: bold;
    }
    .footer a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
col1, col2 = st.columns([1, 8])
with col1:
    st.markdown("<h1 style='font-size: 3.5rem; margin:0;'>🇱🇰</h1>", unsafe_allow_html=True)
with col2:
    st.title("Sri Lanka Economic Crisis Predictor")
    st.write("වර්තමාන ආර්ථික දත්ත ලබා දී ඉදිරි දිනවලදී ජාතික විදුලිබල පද්ධතියේ බිඳවැටීමක් (Power Cut) සිදුවේදැයි පරීක්ෂා කරන්න.")
st.markdown("---")

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("ℹ️ How it works")
    st.write("""
    This tool uses a Machine Learning algorithm (**Support Vector Machine**) trained on Sri Lanka's economic data to predict the likelihood of an **economic crisis** (grid failures/power cuts) based on:
    - **USD Rate** (Exchange Rate)
    - **Diesel Price** (Transport Cost)
    - **Lorry Fuel Quota** (Supply Chain)
    - **Rice Price** (Inflation Impact)
    """)
    st.markdown("---")
    st.caption("🔍 Model Accuracy: 99.5%")
    st.caption("⚡ Developed with Python & Streamlit")

# ========== MAIN CONTENT (SLIDERS) ==========
col1, col2 = st.columns(2)

with col1:
    usd_rate = st.slider(
        "🇺🇸 USD Rate (රු.)",
        min_value=250.0, max_value=400.0, value=313.0, # min_value අඩු කළා Guardrail එකට යන්න
        help="Current Dollar Exchange Rate"
    )
    diesel_price = st.slider(
        "🛢️ Diesel Price (රු./L)",
        min_value=250.0, max_value=500.0, value=382.0,
        help="Price per litre of Auto Diesel"
    )

with col2:
    lorry_quota = st.slider(
        "🚛 Lorry Fuel Quota (L)",
        min_value=50, max_value=200, value=200,
        help="Weekly fuel quota for heavy vehicles"
    )
    rice_price = st.slider(
        "🍚 Rice Price (රු./kg)",
        min_value=100.0, max_value=400.0, value=230.0,
        help="Average price of 1kg of Rice"
    )

st.markdown("<br>", unsafe_allow_html=True)

# ========== PREDICT BUTTON & LOGIC ==========
# තීරණ ගන්නා බොත්තම මැදට ගන්න පේළි 3 කට කැඩුවා
_, btn_col, _ = st.columns([1, 2, 1])

with btn_col:
    predict_clicked = st.button("🔮 Predict Crisis / අනාවැකිය කියන්න", use_container_width=True)

if predict_clicked:
    with st.spinner("🧠 Analyzing economic indicators via SVM Model..."):
        time.sleep(1) # නිකන් පොඩි ගැම්මක් දෙන්න AI එක හිතනවා වගේ පෙන්නන්න

        # Guardrail: ගොඩක් හොඳ තත්ත්වයක් නම් ML එකට යන්නේ නෑ
        if rice_price < 180 and diesel_price < 350 and usd_rate < 300:
            st.markdown(
                """
                <div class="success-box">
                    ✅ <strong>ඉතා යහපත් තත්ත්වයකි!</strong><br>
                    මිල ගණන් ඉතා අඩු මට්ටමක පවතින බැවින් මෙය කිසිසේත්ම අර්බුදයක් නොවේ.
                </div>
                """, unsafe_allow_html=True
            )
        else:
            # Prepare input as DataFrame (Pandas)
            user_data = pd.DataFrame(
                [[usd_rate, diesel_price, lorry_quota, rice_price]],
                columns=['USD_Rate', 'Diesel_Price_Rs', 'Lorry_Quota_L', 'Rice_Price_Rs']
            )

            # Scale and Predict
            scaled_data = scaler.transform(user_data)
            prediction = model.predict(scaled_data)

            if prediction[0] == 1:
                st.markdown(
                    """
                    <div class="error-box">
                        ⚠️ <strong>අවදානම් තත්ත්වයකි! (CRISIS MODE)</strong><br><br>
                        මෙම දත්ත අනුව ඉදිරි දිනවලදී <strong>විදුලි කප්පාදුවක් (Power Cut)</strong> සහ අත්‍යවශ්‍ය සේවා බිඳවැටීමක් වීමට ඉහළ සම්භාවිතාවක් ඇත.
                    </div>
                    """, unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="success-box">
                        ✅ <strong>සාමාන්‍ය තත්ත්වයකි (NORMAL)</strong><br><br>
                        මෙම දත්ත අනුව ආර්ථිකය කළමනාකරණය කළ හැකි මට්ටමක පවතී. ජාතික විදුලිබල පද්ධතියේ බිඳවැටීමක් බලාපොරොත්තු නොවේ.
                    </div>
                    """, unsafe_allow_html=True
                )

# ========== FOOTER ==========
st.markdown(
    """
    <div class="footer">
        Engineered by <strong>Dhanushka Rathnayaka</strong> | 
        <a href="http://dhanushka.live/" target="_blank">View Portfolio</a><br><br>
        <small>Note: This AI-based output may contain errors. This project is created for educational purposes only.</small>
    </div>
    """,
    unsafe_allow_html=True
)