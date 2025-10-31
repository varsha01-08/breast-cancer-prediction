import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# --- Streamlit Page Config ---
st.set_page_config(page_title="🩺 Breast Cancer Detection", page_icon="💖", layout="wide")

# --- App Title ---
st.title("🩺 Breast Cancer Detection App using Machine Learning")
st.markdown("""
Upload your **Breast Cancer dataset (CSV)** and test predictions instantly.  
This app uses **Logistic Regression** to predict whether a tumor is **Benign (Non-cancerous)** or **Malignant (Cancerous)**.
""")

st.divider()

# --- File Upload Section ---
uploaded_file = st.file_uploader("📁 Upload your Breast Cancer Dataset (CSV file)", type=["csv"])

if uploaded_file is not None:
    try:
        # Load Dataset
        data = pd.read_csv(uploaded_file)
        st.success("✅ Dataset loaded successfully!")

        st.subheader("📊 Dataset Preview")
        st.dataframe(data.head())

        # Drop unnecessary columns if present
        if 'id' in data.columns:
            data = data.drop(columns=['id'])
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]  # Drop unnamed cols

        # Identify target column (common names)
        if 'diagnosis' in data.columns:
            y = data['diagnosis']
            X = data.drop(columns=['diagnosis'])
        elif 'target' in data.columns:
            y = data['target']
            X = data.drop(columns=['target'])
        else:
            st.error("❌ Could not find a target column like 'diagnosis' or 'target'.")
            st.stop()

        # Convert categorical target values ('M', 'B') to numeric
        if y.dtype == 'object':
            y = y.map({'M': 1, 'B': 0})

        # Keep only numeric features
        X = X.select_dtypes(include=['number'])

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the Logistic Regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)

        st.subheader("✅ Model Evaluation Results")
        st.write(f"**Accuracy:** {accuracy*100:.2f}%")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        st.text("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))

        st.divider()

        # --- Manual Prediction Section ---
        st.subheader("🧮 Try Your Own Prediction")

        st.markdown("Enter feature values below (you can adjust based on dataset mean values).")

        input_data = []
        col_count = len(X.columns)

        # Create input fields in columns for cleaner UI
        cols = st.columns(3)
        for i, col_name in enumerate(X.columns):
            col_index = i % 3
            with cols[col_index]:
                val = st.number_input(f"{col_name}", value=float(X[col_name].mean()))
                input_data.append(val)

        if st.button("🔍 Predict Cancer Type"):
            input_df = pd.DataFrame([input_data], columns=X.columns)
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)

            st.markdown("---")
            if prediction[0] == 1:
                st.error("🔴 **Result: Malignant (Cancerous)**")
            else:
                st.success("🟢 **Result: Benign (Non-cancerous)**")

    except Exception as e:
        st.error(f"⚠️ Error occurred: {e}")

else:
    st.info("📁 Please upload a CSV file to start.")

st.divider()
st.caption("💖 Developed by Varsha | Streamlit Machine Learning Project © 2025")
