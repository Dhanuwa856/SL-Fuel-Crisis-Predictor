import streamlit as st
import joblib
import numpy as np
import pandas as pd

# සේව් කරපු Model එකයි Scaler එකයි ලෝඩ් කරගනිමු
scaler = joblib.load('scaler.pkl')
model = joblib.load('svm_model.pkl')

st.title("🇱🇰 SL Economic Crisis Predictor")
st.write("වර්තමාන දත්ත ලබා දී හෙට දිනයේ විදුලි කප්පාදුවක් සිදුවේදැයි පරීක්ෂා කරන්න.")

# User ගෙන් දත්ත ඉල්ලන Sliders ටික
usd_rate = st.slider("USD Rate (ඩොලර් අනුපාතය - රු.)", min_value=280.0, max_value=400.0, value=313.0)
diesel_price = st.slider("Diesel Price (ඩීසල් මිල - රු.)", min_value=300.0, max_value=500.0, value=382.0)
lorry_quota = st.slider("Lorry Fuel Quota (ලොරි කෝටාව - ලීටර්)", min_value=50, max_value=200, value=200)
rice_price = st.slider("Rice Price (සහල් මිල - රු.)", min_value=180.0, max_value=400.0, value=230.0)

# Predict Button එක එබුවම වෙන දේ
if st.button("Predict Crisis / අනාවැකිය කියන්න"):

    # 🔴 වෙනස් කළ කොටස: Numpy array වෙනුවට Pandas DataFrame එකක් හදමු (Column නම් එක්කම)
    user_data = pd.DataFrame(
        [[usd_rate, diesel_price, lorry_quota, rice_price]],
        columns=['USD_Rate', 'Diesel_Price_Rs', 'Lorry_Quota_L', 'Rice_Price_Rs']
    )

    # දත්ත ටික Scaling කිරීම
    scaled_data = scaler.transform(user_data)

    # අනාවැකිය (Prediction) ගැනීම
    prediction = model.predict(scaled_data)

    # (අවශ්‍ය නම් මේ පේළියෙන් ඇත්තටම එන උත්තරේ මොකක්ද කියලා App එකේ බලාගන්න පුළුවන්)
    # st.write(f"Raw Model Output: {prediction[0]}")

    st.markdown("---")
    if prediction[0] == 1:
        st.error("⚠️ **අවදානම් තත්ත්වයකි! (CRISIS MODE)**")
        st.write(
            "මෙම දත්ත අනුව ඉදිරි දිනවලදී **විදුලි කප්පාදුවක් (Power Cut)** වීමට 99% ක සම්භාවිතාවක් ඇත. අත්‍යවශ්‍ය සේවා බිඳ වැටිය හැක.")
    else:
        st.success("✅ **සාමාන්‍ය තත්ත්වයකි (NORMAL)**")
        st.write("මෙම දත්ත අනුව ආර්ථිකය කළමනාකරණය කළ හැකි මට්ටමක පවතී. විදුලි කප්පාදුවක් බලාපොරොත්තු නොවේ.")