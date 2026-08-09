import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="Breast Cancer ML Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TEST_DATA_PATH = BASE_DIR / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "kNN": MODEL_DIR / "knn.joblib",
    "Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
    "Gradient Boosting": MODEL_DIR / "gradient_boosting.joblib",
}

MODEL_DESCRIPTIONS = {
    "Logistic Regression": "A strong linear classification baseline with probability estimates.",
    "Decision Tree": "A rule-based non-linear classifier that is easy to interpret.",
    "kNN": "A distance-based classifier that uses nearby observations.",
    "Naive Bayes": "A fast probabilistic classifier based on conditional independence assumptions.",
    "Random Forest": "An ensemble of randomized decision trees designed for robust non-linear learning.",
    "Gradient Boosting": "A sequential ensemble that improves weak learners by focusing on previous errors.",
}


# ============================================================
# Custom styling
# ============================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.22);
        background: linear-gradient(
            135deg,
            rgba(128,128,128,0.10),
            rgba(128,128,128,0.03)
        );
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 750;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.72;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }

    .info-card {
        border: 1px solid rgba(128,128,128,0.20);
        border-radius: 14px;
        padding: 1rem;
        min-height: 110px;
    }

    .footer-note {
        text-align: center;
        opacity: 0.55;
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Model utilities
# ============================================================
@st.cache_resource(show_spinner=False)
def load_artifact(path):
    return joblib.load(str(path))


def available_models():
    return [name for name, path in MODEL_FILES.items() if path.exists()]


def validate_dataset(df, artifact):
    features = artifact["feature_names"]
    target = artifact["target_name"]

    missing = [column for column in features if column not in df.columns]

    if missing:
        return False, (
            "The selected CSV is missing required feature columns: "
            + ", ".join(missing)
        ), None

    if target not in df.columns:
        return False, (
            f"The target column '{target}' is required for model evaluation."
        ), None

    selected = df[features + [target]].copy()

    if selected[features].isnull().any().any():
        return False, "Missing values were detected in the feature columns.", None

    if selected[target].isnull().any():
        return False, "Missing values were detected in the target column.", None

    for feature in features:
        if not pd.api.types.is_numeric_dtype(selected[feature]):
            return False, f"Feature '{feature}' must be numeric.", None

    return True, "", selected


def evaluate(artifact, df):
    model = artifact["model"]
    features = artifact["feature_names"]
    target = artifact["target_name"]

    X = df[features]
    y = df[target].astype(int)

    prediction = model.predict(X)
    probability = model.predict_proba(X)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y, prediction),
        "AUC": roc_auc_score(y, probability),
        "Precision": precision_score(y, prediction, zero_division=0),
        "Recall": recall_score(y, prediction, zero_division=0),
        "F1 Score": f1_score(y, prediction, zero_division=0),
        "MCC": matthews_corrcoef(y, prediction),
    }

    return y, prediction, probability, metrics


# ============================================================
# Header
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🧬 Breast Cancer Classification ML Lab</div>
        <div class="hero-subtitle">
            Interactive evaluation, comparison and prediction dashboard
            for six supervised machine-learning models.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("⚙️ Experiment Controls")

    installed_models = available_models()

    if not installed_models:
        st.error(
            "No model artifacts were found. Upload the complete `model/` "
            "folder containing the six `.joblib` files."
        )
        st.stop()

    selected_model = st.selectbox(
        "Classification Model",
        installed_models,
        help="Select one of the trained models generated by the Colab notebook.",
    )

    st.divider()

    # ========================================================
    # Test Dataset Selection
    # ========================================================
    st.subheader("📂 Test Dataset")

    data_source = st.radio(
        "Choose test-data source",
        options=[
            "Use test_data.csv from repository",
            "Upload another CSV",
        ],
        index=0,
        help=(
            "By default, the app uses the test_data.csv included in the "
            "GitHub repository. You can optionally upload another compatible CSV."
        ),
    )

    if data_source == "Use test_data.csv from repository":

        if TEST_DATA_PATH.exists():
            data = pd.read_csv(TEST_DATA_PATH)
            source_label = "test_data.csv (repository)"

            st.success(
                f"Using repository test data: {len(data)} rows"
            )

        else:
            st.error(
                "test_data.csv was not found in the repository. "
                "Please make sure the file is located beside app.py."
            )
            st.stop()

    else:

        uploaded = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            help=(
                "Upload a compatible test CSV containing the required "
                "features and target column."
            ),
        )

        if uploaded is None:
            st.info(
                "Please upload a CSV file to continue."
            )
            st.stop()

        try:
            data = pd.read_csv(uploaded)
            source_label = uploaded.name

            st.success(
                f"Uploaded dataset: {uploaded.name} • {len(data)} rows"
            )

        except Exception as exc:
            st.error(
                f"Unable to read the uploaded CSV: {exc}"
            )
            st.stop()

    st.divider()

    show_comparison = st.checkbox(
        "Compare all six models",
        value=True,
    )

    show_raw_data = st.checkbox(
        "Show raw test data",
        value=False,
    )

    st.divider()

    st.caption("ML Assignment 2")
    st.caption("Breast Cancer Wisconsin (Diagnostic)")


# ============================================================
# Validate selected model and data
# ============================================================
selected_artifact = load_artifact(
    MODEL_FILES[selected_model]
)

valid, message, data = validate_dataset(
    data,
    selected_artifact,
)

if not valid:
    st.error(message)
    st.stop()


y_true, y_pred, y_prob, selected_metrics = evaluate(
    selected_artifact,
    data,
)


# ============================================================
# Status banner
# ============================================================
st.success(
    f"Active model: **{selected_model}**  •  "
    f"Dataset: **{source_label}**  •  "
    f"Rows evaluated: **{len(data)}**"
)

st.caption(MODEL_DESCRIPTIONS[selected_model])


# ============================================================
# KPI metrics
# ============================================================
st.markdown(
    '<div class="section-title">📊 Model Performance</div>',
    unsafe_allow_html=True,
)

st.write("")

cols = st.columns(6)

for col, (name, value) in zip(
    cols,
    selected_metrics.items(),
):
    col.metric(
        name,
        f"{value:.4f}",
    )


# ============================================================
# All-model benchmark
# ============================================================
if show_comparison:

    st.divider()

    st.markdown(
        '<div class="section-title">🏆 Six-Model Benchmark</div>',
        unsafe_allow_html=True,
    )

    rows = []

    for model_name in installed_models:

        artifact = load_artifact(
            MODEL_FILES[model_name]
        )

        ok, _, model_data = validate_dataset(
            data,
            artifact,
        )

        if not ok:
            continue

        _, _, _, metrics = evaluate(
            artifact,
            model_data,
        )

        rows.append(
            {
                "Model": model_name,
                "Accuracy": metrics["Accuracy"],
                "AUC": metrics["AUC"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1 Score": metrics["F1 Score"],
                "MCC": metrics["MCC"],
            }
        )

    if rows:

        benchmark = pd.DataFrame(rows).sort_values(
            ["F1 Score", "AUC", "MCC"],
            ascending=False,
        ).reset_index(drop=True)

        benchmark.insert(
            0,
            "Rank",
            np.arange(1, len(benchmark) + 1),
        )

        st.dataframe(
            benchmark.style.format(
                {
                    "Accuracy": "{:.4f}",
                    "AUC": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1 Score": "{:.4f}",
                    "MCC": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        leader = benchmark.iloc[0]

        st.info(
            f"**Current test-set leader:** {leader['Model']}  |  "
            f"F1 = {leader['F1 Score']:.4f}  |  "
            f"AUC = {leader['AUC']:.4f}  |  "
            f"MCC = {leader['MCC']:.4f}"
        )

    else:
        st.warning(
            "No models could be evaluated with the selected dataset."
        )


# ============================================================
# Diagnostics
# ============================================================
st.divider()

st.markdown(
    '<div class="section-title">🔎 Model Diagnostics</div>',
    unsafe_allow_html=True,
)

tab_cm, tab_roc, tab_report = st.tabs(
    [
        "Confusion Matrix",
        "ROC Curve",
        "Classification Report",
    ]
)


# ============================================================
# Confusion Matrix
# ============================================================
with tab_cm:

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    fig, ax = plt.subplots(
        figsize=(6.5, 5)
    )

    image = ax.imshow(cm)

    ax.set_title(
        f"Confusion Matrix — {selected_model}"
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks(
        [0, 1],
        ["Malignant", "Benign"],
    )

    ax.set_yticks(
        [0, 1],
        ["Malignant", "Benign"],
    )

    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14,
            )

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    tn, fp, fn, tp = cm.ravel()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "True Negatives",
        tn,
    )

    c2.metric(
        "False Positives",
        fp,
    )

    c3.metric(
        "False Negatives",
        fn,
    )

    c4.metric(
        "True Positives",
        tp,
    )


# ============================================================
# ROC Curve
# ============================================================
with tab_roc:

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob,
    )

    roc_auc = auc(
        fpr,
        tpr,
    )

    fig, ax = plt.subplots(
        figsize=(7.5, 5.2)
    )

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=(
            f"{selected_model} — "
            f"AUC {roc_auc:.4f}"
        ),
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Random classifier",
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curve"
    )

    ax.legend(
        loc="lower right"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# Classification Report
# ============================================================
with tab_report:

    report = classification_report(
        y_true,
        y_pred,
        target_names=[
            "Malignant",
            "Benign",
        ],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(
        report
    ).T

    st.dataframe(
        report_df.style.format(
            {
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1-score": "{:.4f}",
                "support": "{:.0f}",
            }
        ),
        use_container_width=True,
    )


# ============================================================
# Prediction Explorer
# ============================================================
st.divider()

st.markdown(
    '<div class="section-title">🔬 Prediction Explorer</div>',
    unsafe_allow_html=True,
)

prediction_output = data.copy()

prediction_output["Predicted Target"] = y_pred

prediction_output["Predicted Class"] = np.where(
    y_pred == 1,
    "Benign",
    "Malignant",
)

prediction_output["Benign Probability"] = y_prob

prediction_output["Prediction Confidence"] = np.maximum(
    y_prob,
    1 - y_prob,
)

st.dataframe(
    prediction_output.head(50).style.format(
        {
            "Benign Probability": "{:.4f}",
            "Prediction Confidence": "{:.4f}",
        }
    ),
    use_container_width=True,
)

prediction_csv = prediction_output.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "⬇️ Download Predictions CSV",
    data=prediction_csv,
    file_name=(
        selected_model.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + "_predictions.csv"
    ),
    mime="text/csv",
)


# ============================================================
# Project information
# ============================================================
st.divider()

info1, info2, info3 = st.columns(3)

with info1:

    st.markdown(
        """
        <div class="info-card">
        <b>Dataset</b><br>
        Breast Cancer Wisconsin (Diagnostic)<br>
        569 observations • 30 features
        </div>
        """,
        unsafe_allow_html=True,
    )

with info2:

    st.markdown(
        """
        <div class="info-card">
        <b>Classification</b><br>
        Binary classification<br>
        0 = Malignant • 1 = Benign
        </div>
        """,
        unsafe_allow_html=True,
    )

with info3:

    st.markdown(
        """
        <div class="info-card">
        <b>Evaluation</b><br>
        Accuracy • AUC • Precision<br>
        Recall • F1 • MCC
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.expander("ℹ️ About the selected model"):

    st.write(
        MODEL_DESCRIPTIONS[selected_model]
    )

    st.write(
        "The model artifact was trained in the accompanying notebook "
        "and loaded from the repository using joblib."
    )


with st.expander("📘 Dataset Information"):

    st.markdown(
        """
        **Breast Cancer Wisconsin (Diagnostic)**

        - 569 observations
        - 30 numerical features
        - Binary classification
        - `0` = malignant
        - `1` = benign
        - 80/20 stratified train/test split
        - Random state = 42
        """
    )


if show_raw_data:

    st.divider()

    st.subheader("🗃️ Raw Test Data")

    st.dataframe(
        data,
        use_container_width=True,
    )


st.markdown(
    '<div class="footer-note">'
    'ML Assignment 2 • Six-model classification evaluation dashboard'
    '</div>',
    unsafe_allow_html=True,
)
