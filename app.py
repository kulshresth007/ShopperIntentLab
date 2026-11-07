
import hashlib
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="ShopperIntentLab",
    page_icon="🛍️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"


# ==========================================================
# 2. FEATURE DEFINITIONS
# ==========================================================

NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay"
]

CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]

CATEGORY_CODE_FEATURES = [
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType"
]

FEATURE_COLUMNS = (
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
)

TARGET_COLUMN = "Revenue"

REQUIRED_COLUMNS = FEATURE_COLUMNS + [
    TARGET_COLUMN
]


# ==========================================================
# 3. MODEL DEFINITIONS
# ==========================================================

MODEL_FILES = {

    "Logistic Regression":
        "logistic_regression.joblib",

    "Decision Tree":
        "decision_tree.joblib",

    "K-Nearest Neighbors":
        "knn.joblib",

    "Gaussian Naive Bayes":
        "gaussian_naive_bayes.joblib",

    "Support Vector Machine":
        "svm.joblib"
}


# ==========================================================
# 4. LOAD SAVED MODELS
# ==========================================================

@st.cache_resource(show_spinner=False)
def load_models():

    models = {}

    for model_name, filename in MODEL_FILES.items():

        model_path = MODEL_DIR / filename

        if not model_path.exists():

            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        models[model_name] = joblib.load(
            model_path
        )

    return models


# ==========================================================
# 5. NORMALIZE TARGET VARIABLE
# ==========================================================

def normalize_target(target):

    if target.isna().any():

        raise ValueError(
            "Revenue contains missing values."
        )

    if pd.api.types.is_bool_dtype(target):

        return target.astype(bool)

    if pd.api.types.is_numeric_dtype(target):

        values = set(target.unique())

        if values.issubset({0, 1}):

            return target.astype(int).astype(bool)

        raise ValueError(
            "Revenue must contain Boolean values or 0/1."
        )

    target_mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False
    }

    normalized = (
        target.astype(str)
        .str.strip()
        .str.lower()
        .map(target_mapping)
    )

    if normalized.isna().any():

        raise ValueError(
            "Revenue must contain True/False or 0/1."
        )

    return normalized.astype(bool)


# ==========================================================
# 6. VALIDATE AND PREPARE UPLOADED DATA
# ==========================================================

def prepare_uploaded_data(dataframe):

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    if dataframe.empty:

        raise ValueError(
            "Uploaded dataset is empty."
        )

    if dataframe[REQUIRED_COLUMNS].isna().any().any():

        raise ValueError(
            "Uploaded dataset contains missing values."
        )

    X = dataframe[FEATURE_COLUMNS].copy()

    y = normalize_target(
        dataframe[TARGET_COLUMN]
    )

    # Convert numerical features

    for column in NUMERIC_FEATURES:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # Convert categorical identifier features

    for column in CATEGORY_CODE_FEATURES:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

        if X[column].isna().any():

            raise ValueError(
                f"Invalid values in {column}."
            )

        if not np.isfinite(
            X[column].to_numpy(dtype=float)
        ).all():

            raise ValueError(
                f"Non-finite values in {column}."
            )

        if not np.all(
            np.isclose(
                X[column],
                np.round(X[column])
            )
        ):

            raise ValueError(
                f"{column} must contain integer category codes."
            )

        X[column] = X[column].astype(int)

    # Convert categorical text features

    X["Month"] = X["Month"].astype(str)

    X["VisitorType"] = X["VisitorType"].astype(str)

    # Convert Weekend into Boolean format

    if not pd.api.types.is_bool_dtype(X["Weekend"]):

        weekend_mapping = {
            "true": True,
            "false": False,
            "1": True,
            "0": False
        }

        X["Weekend"] = (
            X["Weekend"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(weekend_mapping)
        )

    if X["Weekend"].isna().any():

        raise ValueError(
            "Weekend must contain True/False or 0/1."
        )

    X["Weekend"] = X["Weekend"].astype(bool)

    # Final numerical validation

    if X.isna().any().any():

        raise ValueError(
            "Invalid or missing feature values detected."
        )

    if not np.isfinite(
        X[NUMERIC_FEATURES].to_numpy(dtype=float)
    ).all():

        raise ValueError(
            "Numerical features contain infinite values."
        )

    return X, y


# ==========================================================
# 7. MODEL EVALUATION FUNCTION
# ==========================================================

def evaluate_model(model, X, y_true):

    # Generate predictions

    y_pred = model.predict(X)

    classifier = model.named_steps["classifier"]

    classes = np.asarray(
        classifier.classes_
    )

    positive_indices = np.flatnonzero(
        classes == True
    )

    if len(positive_indices) != 1:

        raise ValueError(
            "Model does not contain the expected positive class."
        )

    positive_index = int(
        positive_indices[0]
    )

    # Obtain continuous prediction scores

    if hasattr(model, "predict_proba"):

        y_scores = model.predict_proba(X)[
            :, positive_index
        ]

    elif hasattr(model, "decision_function"):

        y_scores = model.decision_function(X)

        if positive_index == 0:

            y_scores = -y_scores

    else:

        raise ValueError(
            "Model does not provide prediction scores."
        )

    # Calculate AUC

    if y_true.nunique() == 2:

        auc = roc_auc_score(
            y_true,
            y_scores
        )

    else:

        auc = np.nan

    # Calculate six evaluation metrics

    metrics = {

        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "AUC": auc,

        "Precision": precision_score(
            y_true,
            y_pred,
            pos_label=True,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            pos_label=True,
            zero_division=0
        ),

        "F1": f1_score(
            y_true,
            y_pred,
            pos_label=True,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_true,
            y_pred
        )
    }

    # Confusion matrix

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[False, True]
    )

    # Classification report

    report = classification_report(
        y_true,
        y_pred,
        labels=[False, True],
        target_names=[
            "No Purchase",
            "Purchase"
        ],
        output_dict=True,
        zero_division=0
    )

    return {

        "metrics": metrics,

        "confusion_matrix": cm,

        "classification_report": pd.DataFrame(
            report
        ).transpose(),

        "predictions": y_pred,

        "scores": y_scores

    }


# ==========================================================
# 8. APPLICATION HEADER
# ==========================================================

st.title("🛍️ ShopperIntentLab")

st.subheader(
    "Online Shopping Purchase Intention Prediction"
)

st.write(
    """
    An interactive machine learning application
    comparing five classification algorithms.

    Upload labelled test data, select a model,
    and evaluate its classification performance.
    """
)

st.divider()


# ==========================================================
# 9. PROJECT OVERVIEW
# ==========================================================

st.header("Machine Learning Dashboard")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Classification Models",
    "5"
)

col2.metric(
    "Input Features",
    "17"
)

col3.metric(
    "Evaluation Metrics",
    "6"
)

st.divider()


# ==========================================================
# 10. CSV UPLOAD
# ==========================================================

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Choose your test CSV file",
    type=["csv"],
    help="CSV must contain all 17 input features and Revenue."
)

if uploaded_file is None:

    st.info(
        "Please upload a labelled CSV file to begin."
    )

else:

    file_bytes = uploaded_file.getvalue()

    file_signature = hashlib.sha256(
        file_bytes
    ).hexdigest()

    try:

        uploaded_dataframe = pd.read_csv(
            BytesIO(file_bytes)
        )

        X_uploaded, y_uploaded = (
            prepare_uploaded_data(
                uploaded_dataframe
            )
        )

        st.success(
            "Dataset uploaded and validated successfully!"
        )

        # Dataset summary

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Observations",
            len(X_uploaded)
        )

        col2.metric(
            "Input Features",
            len(FEATURE_COLUMNS)
        )

        col3.metric(
            "Target Classes",
            y_uploaded.nunique()
        )

        # Dataset preview

        with st.expander("Preview Uploaded Dataset"):

            st.dataframe(
                uploaded_dataframe.head(10),
                use_container_width=True
            )

        # Revenue distribution

        st.subheader("Revenue Distribution")

        revenue_counts = (
            y_uploaded.value_counts()
            .rename(
                index={
                    False: "No Purchase",
                    True: "Purchase"
                }
            )
        )

        st.bar_chart(revenue_counts)

        st.divider()


        # ==================================================
        # 11. MODEL SELECTION
        # ==================================================

        st.header("2. Select Classification Model")

        selected_model = st.selectbox(
            "Choose a machine learning classifier:",
            options=list(MODEL_FILES.keys())
        )

        st.info(
            f"Selected Model: {selected_model}"
        )


        # ==================================================
        # 12. SELECTED MODEL EVALUATION
        # ==================================================

        if st.button(
            "Evaluate Selected Model",
            type="primary",
            use_container_width=True
        ):

            try:

                models = load_models()

                model = models[selected_model]

                with st.spinner(
                    f"Evaluating {selected_model}..."
                ):

                    evaluation = evaluate_model(
                        model,
                        X_uploaded,
                        y_uploaded
                    )

                st.session_state[
                    "last_evaluation"
                ] = {

                    "signature": file_signature,

                    "model_name": selected_model,

                    "result": evaluation

                }

                st.success(
                    "Model evaluation completed successfully!"
                )

            except Exception as error:

                st.error(
                    f"Evaluation failed: {error}"
                )


        # ==================================================
        # 13. DISPLAY SELECTED MODEL RESULTS
        # ==================================================

        last_evaluation = st.session_state.get(
            "last_evaluation"
        )

        if (
            last_evaluation is not None
            and last_evaluation["signature"] == file_signature
            and last_evaluation["model_name"] == selected_model
        ):

            result = last_evaluation["result"]

            metrics = result["metrics"]

            st.divider()

            st.header("3. Evaluation Metrics")

            metric_names = [
                "Accuracy",
                "AUC",
                "Precision",
                "Recall",
                "F1",
                "MCC"
            ]

            # Display two rows of three metrics

            for row_start in [0, 3]:

                metric_columns = st.columns(3)

                for index, metric_name in enumerate(
                    metric_names[row_start:row_start + 3]
                ):

                    value = metrics[metric_name]

                    metric_columns[index].metric(
                        metric_name,
                        (
                            f"{value:.4f}"
                            if np.isfinite(value)
                            else "N/A"
                        )
                    )

            # Confusion matrix

            st.divider()

            st.header("4. Confusion Matrix")

            cm_dataframe = pd.DataFrame(

                result["confusion_matrix"],

                index=[
                    "Actual: No Purchase",
                    "Actual: Purchase"
                ],

                columns=[
                    "Predicted: No Purchase",
                    "Predicted: Purchase"
                ]

            )

            st.dataframe(
                cm_dataframe,
                use_container_width=True
            )

            # Classification report

            st.divider()

            st.header("5. Classification Report")

            report_display = (
                result["classification_report"]
                .style.format({
                    "precision": "{:.4f}",
                    "recall": "{:.4f}",
                    "f1-score": "{:.4f}",
                    "support": "{:.0f}"
                })
            )

            st.dataframe(
                report_display,
                use_container_width=True
            )

            # Download prediction results

            st.divider()

            st.header("6. Download Prediction Results")

            prediction_dataframe = uploaded_dataframe.copy()

            prediction_dataframe[
                "Predicted_Revenue"
            ] = result["predictions"]

            st.download_button(

                label="Download Prediction Results",

                data=prediction_dataframe.to_csv(
                    index=False
                ).encode("utf-8"),

                file_name="shopper_predictions.csv",

                mime="text/csv",

                use_container_width=True

            )


        # ==================================================
        # 14. COMPARE ALL FIVE MODELS
        # ==================================================

        st.divider()

        st.header("7. Compare All Five Models")

        if st.button(
            "Evaluate All Five Models",
            use_container_width=True
        ):

            try:

                models = load_models()

                all_results = {}

                progress = st.progress(0)

                status_text = st.empty()

                total_models = len(models)

                for index, (name, model) in enumerate(
                    models.items()
                ):

                    status_text.write(
                        f"Evaluating {name}..."
                    )

                    all_results[name] = evaluate_model(
                        model,
                        X_uploaded,
                        y_uploaded
                    )

                    progress.progress(
                        (index + 1) / total_models
                    )

                st.session_state[
                    "all_model_results"
                ] = {

                    "signature": file_signature,

                    "results": all_results

                }

                status_text.empty()

                progress.empty()

                st.success(
                    "All five models evaluated successfully!"
                )

            except Exception as error:

                st.error(
                    f"Comparison failed: {error}"
                )


        # ==================================================
        # 15. DISPLAY FIVE-MODEL COMPARISON
        # ==================================================

        stored_comparison = st.session_state.get(
            "all_model_results"
        )

        if (
            stored_comparison is not None
            and stored_comparison["signature"] == file_signature
        ):

            comparison_dataframe = pd.DataFrame({

                name: result["metrics"]

                for name, result
                in stored_comparison["results"].items()

            }).transpose()

            st.subheader(
                "Five-Model Performance Comparison"
            )

            # Display exactly four decimal places

            comparison_display = (
                comparison_dataframe.style.format(
                    "{:.4f}"
                )
            )

            st.dataframe(
                comparison_display,
                use_container_width=True
            )

            # Prepare clean CSV with exactly four decimals

            comparison_export = (
                comparison_dataframe
                .reset_index()
                .rename(columns={"index": "Model"})
            )

            for column in [
                "Accuracy",
                "AUC",
                "Precision",
                "Recall",
                "F1",
                "MCC"
            ]:

                comparison_export[column] = (
                    comparison_export[column]
                    .map(
                        lambda value: (
                            f"{value:.4f}"
                            if pd.notna(value)
                            else ""
                        )
                    )
                )

            st.download_button(

                label="Download Comparison Results",

                data=comparison_export.to_csv(
                    index=False
                ).encode("utf-8"),

                file_name="five_model_comparison.csv",

                mime="text/csv"

            )

    except Exception as error:

        st.error(
            f"Dataset validation failed: {error}"
        )


# ==========================================================
# 16. FOOTER
# ==========================================================

st.divider()

st.caption(
    "ShopperIntentLab | BITS Pilani | M.Tech AIML"
)
