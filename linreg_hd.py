import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------- STYLING ----
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #e0f2fe 0%, #f5f3ff 50%, #fce7f3 100%);
    }
    #MainMenu, footer, header {visibility: hidden;}

    /* Remove the big empty space at the top so no scrolling is needed */
    .block-container {padding-top: 1rem; padding-bottom: 0.5rem; max-width: 1400px;}

    /* Compact hero banner */
    .hero {
        background: linear-gradient(120deg, #4f46e5, #7c3aed, #db2777);
        padding: 0.9rem 1.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.35);
        margin-bottom: 0.8rem;
    }
    .hero h1 {margin: 0; font-size: 1.9rem; color: white; padding: 0;}
    .hero p  {margin: 0.1rem 0 0; font-size: 0.95rem; opacity: 0.92; color: white;}

    /* Cards (st.container with border=True) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.88);
        border-radius: 16px;
        border: none;
        border-left: 6px solid #7c3aed;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }
    .card-title {font-size: 1.15rem; font-weight: 700; color: #4338ca; margin-bottom: 0.2rem;}

    /* Labels */
    label p {font-weight: 600 !important; color: #1e293b !important;}

    /* Predict button */
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
        padding: 1.6rem 1rem;
        border-radius: 18px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 28px rgba(16, 185, 129, 0.4);
        margin-top: 0.8rem;
    }
    .result.waiting {background: linear-gradient(120deg, #94a3b8, #64748b); box-shadow: none;}
    .result .label {font-size: 1rem; opacity: 0.92;}
    .result .value {font-size: 2.6rem; font-weight: 800; margin: 0.2rem 0;}
    .result .hint  {font-size: 0.85rem; opacity: 0.85;}

    section[data-testid="stSidebar"] {background: linear-gradient(180deg, #312e81, #6d28d9);}
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


def pretty(name):
    return name.replace("_", " ")


# ---------------------------------------------------------------- SIDEBAR ----
with st.sidebar:
    st.markdown("## 📊 About")
    st.write("Linear Regression model built with scikit-learn.")
    st.markdown("**Features used**")
    for f in features:
        st.write(f"• {pretty(f)}")

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

# ----------------------------------------------------------------- LAYOUT ----
left, right = st.columns([2.2, 1], gap="large")

inputs = {}

# ---- LEFT: all inputs in a 3-column grid -----------------------------------
with left:
    with st.container(border=True):
        st.markdown('<div class="card-title">🏡 Property Details</div>', unsafe_allow_html=True)

        items = list(other_cols)
        if region_cols:
            items.append("__region__")

        grid = st.columns(3)
        for i, col in enumerate(items):
            with grid[i % 3]:
                name = col.lower()

                if col == "__region__":
                    baseline = "Other (baseline region)"  # region dropped by drop_first=True
                    options = [baseline] + [c.replace("Region_", "") for c in region_cols]
                    region = st.selectbox("📍 Region", options, key="in_region")
                    for c in region_cols:
                        inputs[c] = 1 if c == f"Region_{region}" else 0

                elif name == "parking":
                    choice = st.selectbox("🚗 Parking", ["One", "Two"], key=f"in_{col}")
                    inputs[col] = 2 if choice == "Two" else 1

                elif "school" in name and "rating" in name:
                    # School rating must be between 1 and 10
                    inputs[col] = float(
                        st.slider(
                            "🎓 School Rating (1-10)",
                            min_value=1, max_value=10, value=5, step=1,
                            key=f"in_{col}",
                        )
                    )

                elif "transport" in name:
                    # Transport: 0 = Yes, 1 = No
                    choice = st.selectbox("🚌 Transport", ["Yes", "No"], key=f"in_{col}")
                    inputs[col] = 0 if choice == "Yes" else 1

                else:
                    # Any other numeric feature: cannot be negative
                    inputs[col] = st.number_input(
                        pretty(col), min_value=0.0, value=0.0, step=1.0, key=f"in_{col}"
                    )

# ---- RIGHT: predict button + result ----------------------------------------
with right:
    with st.container(border=True):
        st.markdown('<div class="card-title">🔮 Prediction</div>', unsafe_allow_html=True)
        go = st.button("Predict Price")

        if go:
            X = pd.DataFrame([inputs])[features]
            pred = model.predict(X)[0]
            st.markdown(
                f"""
                <div class="result">
                    <div class="label">Estimated Value(in Lakhs)</div>
                    <div class="value">{pred:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.balloons()
        else:
            st.markdown(
                """
                <div class="result waiting">
                    <div class="label">Estimated Value(in Lakhs)</div>
                    <div class="value">—</div>
                    <div class="hint">Enter the details and click Predict</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Feature details: the exact values being sent to the model
        with st.expander("🔍 Feature details (input sent to the model)"):
            details = pd.DataFrame([inputs])[features].T
            details.columns = ["Value"]
            details.index.name = "Feature"
            st.dataframe(details)