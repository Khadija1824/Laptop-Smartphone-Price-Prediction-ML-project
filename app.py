import streamlit as st
import pandas as pd
import pickle
import time

# Page Configuration - Full-width layout
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# Custom CSS for removing top whitespace, adding aesthetic header, and wider layout
st.markdown("""
    <style>
    /* Completely remove Streamlit top header bar and default whitespace */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 3rem !important;
        max-width: 950px !important;
    }

    /* Full page background styling */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }

    /* Aesthetic Header Box with Gradient Background */
    .main-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2.5rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #818cf8, #c084fc, #e879f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1.15rem;
        margin: 0;
    }

    /* Section Subheadings */
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-top: 2rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 0.6rem;
    }

    /* Streamlit Input Styling for Dark Theme */
    label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
    }
    
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background-color: rgba(30, 41, 59, 0.7) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
    }

    /* Custom Glowing Button Color */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
        transform: translateY(-2px);
    }

    /* Smooth Fade-In Animation for Result */
    @keyframes fadeInScale {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }

    .result-container {
        animation: fadeInScale 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%);
        padding: 2rem;
        border-radius: 14px;
        text-align: center;
        color: #ffffff;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-top: 1.5rem;
    }
    
    .result-container h3 {
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 0;
        opacity: 0.9;
    }
    
    .result-container h1 {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0 0 0;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Load trained model pipeline safely
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Model file `model.pkl` not found! Please run `python3 train_model.py` first.")
    st.stop()

# --- AESTHETIC BACKGROUND HEADER BOX ---
st.markdown("""
    <div class="main-header">
        <h1>💻 Laptop Price Predictor Dashboard</h1>
        <p>Estimate advanced hardware market valuations instantly using Machine Learning</p>
    </div>
""", unsafe_allow_html=True)

# --- WIDER SECTION LAYOUT ---

# 1. Configure Hardware Specifications Section
st.markdown('<div class="section-title">⚙️ Configure Hardware Specifications</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    company = st.selectbox('Brand (Company)', ['Apple', 'HP', 'Dell', 'Asus', 'Acer', 'Lenovo'])
    type_name = st.selectbox('Laptop Type', ['Ultrabook', 'Notebook', 'Gaming'])
    inches = st.number_input('Screen Size (Inches)', min_value=10.0, max_value=20.0, value=15.6, step=0.1)

with col2:
    ram = st.selectbox('RAM (GB)', [4, 8, 16, 32])
    memory = st.selectbox('Storage (Memory)', ['128GB SSD', '256GB SSD', '512GB SSD', '1TB SSD', '500GB HDD', '1TB HDD'])
    weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, value=2.0, step=0.1)

# 2. Valuation & Real-Time Analysis Section
st.markdown('<div class="section-title">📊 Valuation & Real-time Analysis</div>', unsafe_allow_html=True)

st.write("Process your hardware specifications through the Random Forest regression engine to calculate the predicted retail price.")

predict_clicked = st.button("🔮 Run Price Prediction Engine", use_container_width=True)

# Prediction Logic with Spinner Transition
if predict_clicked:
    input_data = pd.DataFrame({
        'Company': [company],
        'TypeName': [type_name],
        'Inches': [inches],
        'Ram': [ram],
        'Memory': [memory],
        'Weight': [weight]
    })
    
    with st.spinner("Analyzing architectural features..."):
        time.sleep(0.4)  # Smooth transition feel
        prediction = model.predict(input_data)[0]

    # Display Aesthetic Result Box
    st.markdown(f"""
        <div class="result-container">
            <h3>Estimated Market Price</h3>
            <h1>₹ {prediction:,.2f}</h1>
        </div>
    """, unsafe_allow_html=True)