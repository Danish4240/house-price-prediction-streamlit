import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

# ---------------------------------------------------------------- STYLING ----
st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background: linear-gradient(135deg, #e0f2fe 0%, #f5f3ff 50%, #fce7f3 100%);
    }

    /* Hide default Streamlit menu/footer for a cleaner look */
    #MainMenu, footer {visibility: hidden;}

    /* Hero banner */
    .hero {
        background: linear-gradient(120deg, #4f46e5, #7c3aed, #db2777);
        padding: 2.2rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.35);
        margin-bottom: 1.5rem;
    }
    .hero h1 {margin: 0; font-size: 2.6rem; color: white;}
    .hero p  {margin: 0.4rem 0 0; font-size: 1.1rem; opacity: 0.92; color: white;}

    /* Section cards */
    .card {
        background: rgba(255, 255, 255, 0.85);
        padding: 1.2rem 1.5rem 0.4rem;
        border-radius: 16px;
        border-left: 6px solid #7c3aed;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        margin-bottom: 1rem;
    }
    .card h3 {margin: 0 0 0.6rem; color: #4338ca;}

    /* Input labels */
    label, .stSelectbox label, .stNumberInput label {
        font-weight: 600 !important;
        color: #1e293b !important;
    }

    /* Main predict button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #f97316, #ec4899);
        color: white;
        font-size: 1.2rem;
        font-weight: 700;
        padding: 0.8rem 1rem;
        border: none;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(236, 72, 153, 0.45);
        transition: all 0.25s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 24px rgba(236, 72, 153, 0.6);
        color: white;
    }
    div.stButton > button:active {transform: scale(0.98);}

    /* Result card */
    .result {
        background: linear-gradient(120deg, #10b981, #06b6d4);
        padding: 1.8rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 28px rgba(16, 185, 129, 0.4);
        margin-top: 1rem;
    }
    .result .label {font-size: 1rem; opacity: 0.9;}
    .result .value {font-size: 3rem; font-weight: 800; margin: 0.2rem 0;}

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #312e81, #6d28d9);
    }
    section[data-testid="stSidebar"] * {color: white !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ MODEL ----
@st.cache_resource
def load_model():
    return joblib.load("linreg_hd.pkl")


model = load_model()
features = list(model.feature_names_in_)
region_cols = [c for c in features if c.startswith("Region_")]
other_cols = [c for c in features if c not in region_cols]

# ---------------------------------------------------------------- SIDEBAR ----
with st.sidebar:
    st.markdown("## 📊 About")
    st.write("Machine learning model built with **Linear Regression** (scikit-learn).")
    st.markdown("---")
    st.markdown("**Features used**")
    for f in features:
        st.write(f"• {f.replace('_', ' ')}")
    st.markdown("---")
    st.caption("Built with Python · Pandas · scikit-learn · Streamlit")

# ------------------------------------------------------------------- HERO ----
st.markdown(
    """
    <div class="hero">
        <h1>🏠 House Price Predictor</h1>
        <p>Fill in the property details and get an instant estimate</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------- INPUTS ----
inputs = {}

left, right = st.columns(2, gap="large")

with left:
    st.markdown('<div class="card"><h3>🏡 Property Details</h3>', unsafe_allow_html=True)
    for col in other_cols:
        if col == "Parking":
            choice = st.selectbox("🚗 Parking", ["One", "Two"])
            inputs[col] = 2 if choice == "Two" else 1
        else:
            inputs[col] = st.number_input(col.replace("_", " "), value=0.0, step=1.0)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card"><h3>📍 Location</h3>', unsafe_allow_html=True)
    if region_cols:
        baseline = "Other (baseline region)"  # region dropped by drop_first=True
        options = [baseline] + [c.replace("Region_", "") for c in region_cols]
        region = st.selectbox("Region", options)
        for c in region_cols:
            inputs[c] = 1 if c == f"Region_{region}" else 0
    else:
        st.info("No region information in this model.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- PREDICT ----
_, mid, _ = st.columns([1, 2, 1])
with mid:
    go = st.button("🔮 Predict Price")

if go:
    X = pd.DataFrame([inputs])[features]
    pred = model.predict(X)[0]
    st.markdown(
        f"""
        <div class="result">
            <div class="label">predicted Value (in Lakhs)</div>
            <div class="value">{pred:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.balloons()
    with st.expander("🔍 Show input data sent to the model"):
        st.dataframe(X, use_container_width=True)