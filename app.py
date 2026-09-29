
import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# Page configuration
# -----------------------------------
st.set_page_config(
    page_title="SAH – DCI Prediction",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------------
# Load trained model
# -----------------------------------
model = joblib.load("sah_dci_model.joblib")

preprocessor = model.named_steps["preprocessor"]

# Get original numerical and categorical columns
numeric_cols = []
categorical_cols = []

for name, transformer, columns in preprocessor.transformers_:

    if name == "num":
        numeric_cols = list(columns)

    elif name == "cat":
        categorical_cols = list(columns)

# -----------------------------------
# Title
# -----------------------------------
st.title("🧠 SAH – DCI Prediction Model")

st.subheader(
    "Admission-based prediction of Delayed Cerebral Ischemia"
)

st.info(
    "Enter the patient's admission variables to estimate "
    "the predicted probability of DCI."
)

st.divider()

# -----------------------------------
# Patient inputs
# -----------------------------------
st.header("Patient Admission Data")

patient_data = {}

# Numerical variables
for col in numeric_cols:

    st.subheader(str(col))

    # Get median used by the trained preprocessing pipeline
    num_transformer = preprocessor.named_transformers_["num"]
    imputer = num_transformer.named_steps["imputer"]

    col_index = list(numeric_cols).index(col)
    default_value = float(imputer.statistics_[col_index])

    patient_data[col] = st.number_input(
        str(col),
        value=default_value
    )

# Categorical variables
for col in categorical_cols:

    cat_transformer = preprocessor.named_transformers_["cat"]
    encoder = cat_transformer.named_steps["onehot"]

    col_index = list(categorical_cols).index(col)

    options = list(encoder.categories_[col_index])

    patient_data[col] = st.selectbox(
        str(col),
        options
    )

st.divider()

# -----------------------------------
# Prediction
# -----------------------------------
if st.button(
    "🔮 Predict DCI Probability",
    use_container_width=True
):

    patient_df = pd.DataFrame([patient_data])

    probability = model.predict_proba(patient_df)[0, 1]

    percentage = probability * 100

    st.subheader("Predicted DCI Probability")

    st.metric(
        label="Probability of Delayed Cerebral Ischemia",
        value=f"{percentage:.1f}%"
    )

    st.progress(float(probability))

    if probability < 0.25:
        st.info("Model-estimated probability: relatively low range.")

    elif probability < 0.50:
        st.warning("Model-estimated probability: intermediate range.")

    else:
        st.error("Model-estimated probability: higher range.")

    st.caption(
        "Research prototype only. This model has not been clinically "
        "validated and should not be used to make patient-management decisions."
    )
