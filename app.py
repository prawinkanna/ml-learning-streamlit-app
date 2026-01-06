import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report
)

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="ML Learning Web App for Freshers",
    layout="wide"
)

st.title("📊 End-to-End Machine Learning Learning App")
st.write("Designed for **freshers to understand ML step by step** 🚀")

# =============================
# INTRO
# =============================
with st.expander("📘 About This Application"):
    st.write(
        "This application is designed for beginners to learn Machine Learning. "
        "It explains each step of the ML workflow including data preprocessing, "
        "model selection, training, and evaluation in simple language."
    )

# =============================
# Step 1: Upload
# =============================
st.subheader("📁 Step 1: Upload Dataset")

with st.expander("ℹ️ Why upload a CSV file?"):
    st.write(
        "Machine Learning works on data. Uploading a CSV file allows the model "
        "to learn patterns from real-world datasets."
    )

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # =============================
        # Step 2: Preview
        # =============================
        df = pd.read_csv(uploaded_file)

        st.subheader("🔍 Step 2: Dataset Preview")
        st.dataframe(df.head())

        c1, c2 = st.columns(2)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])

        # =============================
        # Step 3: Missing Values
        # =============================
        st.subheader("🚨 Step 3: Missing Value Check")

        with st.expander("ℹ️ Why check missing values?"):
            st.write(
                "Missing values can cause errors or incorrect predictions. "
                "Most ML models cannot work with empty (null) values, "
                "so they must be handled before training."
            )

        st.dataframe(df.isnull().sum())

        # =============================
        # Step 4: Data Cleaning
        # =============================
        st.subheader("🧹 Step 4: Data Cleaning")

        with st.expander("ℹ️ What is data cleaning?"):
            st.write(
                "Data cleaning is the process of fixing or removing incorrect, "
                "missing, or unnecessary data to improve model performance."
            )

        clean_df = df.copy()

        method = st.selectbox(
            "Choose missing value handling method",
            ["Do Nothing", "Mean", "Median", "Mode", "Drop Rows"]
        )

        with st.expander("ℹ️ Explanation of cleaning methods"):
            st.write(
                "• Mean: Replaces missing values with the average value\n"
                "• Median: Replaces with middle value (robust to outliers)\n"
                "• Mode: Replaces with most frequent value\n"
                "• Drop Rows: Removes rows with missing values"
            )

        if method == "Mean":
            clean_df = clean_df.fillna(clean_df.mean(numeric_only=True))
        elif method == "Median":
            clean_df = clean_df.fillna(clean_df.median(numeric_only=True))
        elif method == "Mode":
            clean_df = clean_df.fillna(clean_df.mode().iloc[0])
        elif method == "Drop Rows":
            clean_df = clean_df.dropna()

        drop_cols = st.multiselect("Drop unnecessary columns", clean_df.columns)

        with st.expander("ℹ️ Why drop columns?"):
            st.write(
                "Some columns like ID, serial number, or irrelevant fields "
                "do not help prediction and should be removed."
            )

        if drop_cols:
            clean_df = clean_df.drop(columns=drop_cols)

        st.dataframe(clean_df.head())

        # =============================
        # Step 5: Target Selection
        # =============================
        st.subheader("🎯 Step 5: Target Column Selection")

        with st.expander("ℹ️ What is a target column?"):
            st.write(
                "The target column is the output that the model tries to predict. "
                "Example: Purchased (Yes/No), Sales amount, Price."
            )

        target_col = st.selectbox("Select target column", clean_df.columns)

        X = clean_df.drop(columns=[target_col])
        y = clean_df[target_col]

        # =============================
        # Step 6: Task Detection
        # =============================
        if y.dtype == "object" or y.nunique() <= 10:
            task_type = "Classification"
        else:
            task_type = "Regression"

        st.success(f"Detected ML Task Type: {task_type}")

        X = pd.get_dummies(X, drop_first=True)

        if task_type == "Classification" and y.dtype == "object":
            y = LabelEncoder().fit_transform(y)

        # =============================
        # Step 7: Train-Test Split
        # =============================
        st.subheader("🔀 Step 7: Train-Test Split")

        with st.expander("ℹ️ Why split data into train and test?"):
            st.write(
                "Training data is used to teach the model, while test data "
                "is used to evaluate how well the model performs on unseen data."
            )

        test_size = st.slider("Test size (%)", 10, 40, 20) / 100

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # =============================
        # Step 8: Model Selection
        # =============================
        st.subheader("🧠 Step 8: Model Selection")

        with st.expander("ℹ️ Explanation of models"):
            st.write(
                "Linear Regression: Simple model for numeric prediction\n"
                "Decision Tree: Rule-based model, easy to understand\n"
                "Random Forest: Multiple trees combined for better accuracy\n"
                "Gradient Boosting: Sequential trees improving previous errors\n"
                "Logistic Regression: Linear classifier for Yes/No problems\n"
                "KNN: Classifies based on nearest data points"
            )

        if task_type == "Regression":
            model_name = st.selectbox(
                "Choose Regression Model",
                [
                    "Linear Regression",
                    "Decision Tree Regressor",
                    "Random Forest Regressor",
                    "Gradient Boosting Regressor"
                ]
            )

            model = {
                "Linear Regression": LinearRegression(),
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
                "Random Forest Regressor": RandomForestRegressor(random_state=42),
                "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
            }[model_name]

        else:
            model_name = st.selectbox(
                "Choose Classification Model",
                [
                    "Logistic Regression",
                    "Decision Tree Classifier",
                    "Random Forest Classifier",
                    "KNN Classifier"
                ]
            )

            if model_name == "KNN Classifier":
                k = st.slider("Select K value", 3, 15, 5)
                model = KNeighborsClassifier(n_neighbors=k)
            else:
                model = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
                    "Random Forest Classifier": RandomForestClassifier(random_state=42)
                }[model_name]

        # =============================
        # Step 9: Training
        # =============================
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # =============================
        # Step 10: Model Evaluation
        # =============================
        st.subheader("📈 Step 10: Model Evaluation")

        with st.expander("ℹ️ What do these metrics mean?"):
            st.write(
                "MAE: Average prediction error\n"
                "RMSE: Penalizes large errors\n"
                "R²: How well the model explains the data\n"
                "Accuracy: Percentage of correct predictions"
            )

        if task_type == "Regression":
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            r2 = r2_score(y_test, y_pred)

            st.metric("MAE", round(mae, 2))
            st.metric("RMSE", round(rmse, 2))
            st.metric("R² Score", round(r2, 2))

        else:
            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy (%)", round(acc * 100, 2))
            st.text(classification_report(y_test, y_pred))

        st.success("✅ Machine Learning workflow completed successfully!")

    except Exception as e:
        st.error("⚠️ Error occurred")
        st.exception(e)

else:
    st.info("👆 Upload a CSV file to begin")
