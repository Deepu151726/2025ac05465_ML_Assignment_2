import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Breast Cancer Classification ML Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_DATA_PATH = BASE_DIR / "train_data.csv"
TEST_DATA_PATH = BASE_DIR / "test_data.csv"

TARGET_COLUMN = "target"
RANDOM_STATE = 42


CLASS_NAMES = {
    0: "Malignant",
    1: "Benign"
}


# ============================================================
# TITLE
# ============================================================

st.title("🧬 Breast Cancer Classification ML Lab")

st.caption(
    "Six-model classification benchmark with interactive "
    "evaluation, diagnostics and prediction analysis."
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


def validate_dataset(df):

    if TARGET_COLUMN not in df.columns:
        return False, (
            f"The dataset must contain the target column "
            f"'{TARGET_COLUMN}'."
        )

    feature_columns = [
        column for column in df.columns
        if column != TARGET_COLUMN
    ]

    if len(feature_columns) != 30:
        return False, (
            f"Expected 30 feature columns, but found "
            f"{len(feature_columns)}."
        )

    if df[feature_columns].isnull().any().any():
        return False, (
            "Missing values were detected in the feature columns."
        )

    if df[TARGET_COLUMN].isnull().any():
        return False, (
            "Missing values were detected in the target column."
        )

    return True, ""


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Experiment Controls")

    st.subheader("🤖 Model Selection")

    model_name = st.selectbox(
        "Choose a classification model",
        [
            "Logistic Regression",
            "Decision Tree",
            "kNN",
            "Naive Bayes",
            "Random Forest",
            "Gradient Boosting"
        ]
    )

    st.divider()

    st.subheader("📂 Test Data")

    data_source = st.radio(
        "Select test-data source",
        [
            "Use test_data.csv from repository",
            "Upload another CSV"
        ],
        index=0
    )

    # --------------------------------------------------------
    # Repository Test Data
    # --------------------------------------------------------

    if data_source == "Use test_data.csv from repository":

        if TEST_DATA_PATH.exists():

            test_df = load_csv(TEST_DATA_PATH)

            st.success(
                f"Repository test data loaded\n\n"
                f"{len(test_df)} observations"
            )

        else:

            st.error(
                "test_data.csv was not found in the repository."
            )

            st.stop()

    # --------------------------------------------------------
    # Uploaded Test Data
    # --------------------------------------------------------

    else:

        uploaded_file = st.file_uploader(
            "Upload test CSV",
            type=["csv"],
            help=(
                "Upload a compatible test dataset containing "
                "30 numerical features and the target column."
            )
        )

        if uploaded_file is None:

            st.info(
                "Upload a CSV file to evaluate the models."
            )

            st.stop()

        try:

            test_df = pd.read_csv(uploaded_file)

        except Exception as error:

            st.error(
                f"Unable to read the CSV file: {error}"
            )

            st.stop()

    st.divider()

    compare_models = st.checkbox(
        "Compare all six models",
        value=True
    )

    show_raw_data = st.checkbox(
        "Show test data",
        value=False
    )


# ============================================================
# VALIDATE TEST DATA
# ============================================================

valid, validation_message = validate_dataset(test_df)

if not valid:

    st.error(validation_message)

    st.stop()


# ============================================================
# LOAD TRAINING DATA
# ============================================================

if not TRAIN_DATA_PATH.exists():

    st.error(
        "train_data.csv was not found in the repository."
    )

    st.stop()


train_df = load_csv(TRAIN_DATA_PATH)


train_valid, train_message = validate_dataset(train_df)

if not train_valid:

    st.error(
        f"Training dataset validation failed: {train_message}"
    )

    st.stop()


# ============================================================
# PREPARE TRAINING AND TEST DATA
# ============================================================

feature_columns = [
    column
    for column in train_df.columns
    if column != TARGET_COLUMN
]


X_train = train_df[feature_columns]
y_train = train_df[TARGET_COLUMN].astype(int)

X_test = test_df[feature_columns]
y_test = test_df[TARGET_COLUMN].astype(int)


# ============================================================
# MODEL FACTORY
# ============================================================

def create_models():

    models = {

        "Logistic Regression": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=RANDOM_STATE
                    )
                )
            ]
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=3,
            random_state=RANDOM_STATE
        ),

        "kNN": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=7
                    )
                )
            ]
        ),

        "Naive Bayes": GaussianNB(),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        )
    }

    return models


# ============================================================
# MODEL TRAINING
# ============================================================

@st.cache_resource
def train_model(model_name, X_train, y_train):

    models = create_models()

    model = models[model_name]

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    metrics = {

        "Accuracy": accuracy_score(
            y,
            predictions
        ),

        "AUC": roc_auc_score(
            y,
            probabilities
        ),

        "Precision": precision_score(
            y,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y,
            predictions,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y,
            predictions,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y,
            predictions
        )
    }

    return (
        predictions,
        probabilities,
        metrics
    )


# ============================================================
# TRAIN SELECTED MODEL
# ============================================================

with st.spinner(
    f"Training {model_name}..."
):

    selected_model = train_model(
        model_name,
        X_train,
        y_train
    )


y_pred, y_prob, selected_metrics = evaluate_model(
    selected_model,
    X_test,
    y_test
)


# ============================================================
# DATASET SUMMARY
# ============================================================

st.success(
    f"Active model: **{model_name}**  |  "
    f"Training samples: **{len(X_train)}**  |  "
    f"Test samples: **{len(X_test)}**"
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.subheader("📋 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Training Samples",
    len(X_train)
)

col2.metric(
    "Test Samples",
    len(X_test)
)

col3.metric(
    "Features",
    len(feature_columns)
)

col4.metric(
    "Target Classes",
    y_train.nunique()
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.subheader("📊 Model Performance")

metric_columns = st.columns(6)

for column, (metric_name, value) in zip(
    metric_columns,
    selected_metrics.items()
):

    column.metric(
        metric_name,
        f"{value:.4f}"
    )


# ============================================================
# SIX MODEL BENCHMARK
# ============================================================

if compare_models:

    st.divider()

    st.subheader("🏆 Six-Model Benchmark")

    all_results = []

    models = create_models()

    progress = st.progress(
        0,
        text="Training six classification models..."
    )

    total_models = len(models)

    for index, (name, model) in enumerate(
        models.items(),
        start=1
    ):

        model.fit(
            X_train,
            y_train
        )

        predictions, probabilities, metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        all_results.append(
            {
                "ML Model Name": name,
                "Accuracy": metrics["Accuracy"],
                "AUC": metrics["AUC"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F1 Score"],
                "MCC": metrics["MCC"]
            }
        )

        progress.progress(
            index / total_models,
            text=f"Evaluated {name}"
        )

    progress.empty()

    benchmark_df = pd.DataFrame(
        all_results
    )

    benchmark_df = benchmark_df.sort_values(
        by=[
            "F1",
            "AUC",
            "MCC"
        ],
        ascending=False
    ).reset_index(drop=True)

    benchmark_df.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(benchmark_df) + 1
        )
    )

    st.dataframe(
        benchmark_df.style.format(
            {
                "Accuracy": "{:.4f}",
                "AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "MCC": "{:.4f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    winner = benchmark_df.iloc[0]

    st.success(
        f"🏆 Best performing model: **{winner['ML Model Name']}**  |  "
        f"Accuracy: **{winner['Accuracy']:.4f}**  |  "
        f"AUC: **{winner['AUC']:.4f}**  |  "
        f"F1: **{winner['F1']:.4f}**  |  "
        f"MCC: **{winner['MCC']:.4f}**"
    )


# ============================================================
# MODEL DIAGNOSTICS
# ============================================================

st.divider()

st.subheader("🔎 Model Diagnostics")

tab1, tab2, tab3 = st.tabs(
    [
        "Confusion Matrix",
        "ROC Curve",
        "Classification Report"
    ]
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

with tab1:

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(6.5, 5)
    )

    image = ax.imshow(cm)

    ax.set_title(
        f"Confusion Matrix — {model_name}"
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_xticks(
        [0, 1],
        ["Malignant", "Benign"]
    )

    ax.set_yticks(
        [0, 1],
        ["Malignant", "Benign"]
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=15
            )

    fig.colorbar(
        image,
        ax=ax
    )

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# ROC CURVE
# ============================================================

with tab2:

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test,
        y_prob
    )

    roc_auc = auc(
        false_positive_rate,
        true_positive_rate
    )

    fig, ax = plt.subplots(
        figsize=(7.5, 5)
    )

    ax.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=f"{model_name} — AUC {roc_auc:.4f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Random classifier"
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
# CLASSIFICATION REPORT
# ============================================================

with tab3:

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Malignant",
            "Benign"
        ],
        output_dict=True,
        zero_division=0
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
                "support": "{:.0f}"
            }
        ),
        use_container_width=True
    )


# ============================================================
# PREDICTION EXPLORER
# ============================================================

st.divider()

st.subheader("🔬 Prediction Explorer")

prediction_output = test_df.copy()

prediction_output["Predicted Target"] = y_pred

prediction_output["Predicted Class"] = [
    CLASS_NAMES[value]
    for value in y_pred
]

prediction_output["Benign Probability"] = y_prob

prediction_output["Prediction Confidence"] = np.maximum(
    y_prob,
    1 - y_prob
)


st.dataframe(
    prediction_output.head(50).style.format(
        {
            "Benign Probability": "{:.4f}",
            "Prediction Confidence": "{:.4f}"
        }
    ),
    use_container_width=True
)


# ============================================================
# DOWNLOAD PREDICTIONS
# ============================================================

prediction_csv = prediction_output.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    "⬇️ Download Predictions",
    data=prediction_csv,
    file_name=(
        model_name.lower()
        .replace(" ", "_")
        + "_predictions.csv"
    ),
    mime="text/csv"
)


# ============================================================
# RAW TEST DATA
# ============================================================

if show_raw_data:

    st.divider()

    st.subheader("🗃️ Test Data")

    st.dataframe(
        test_df,
        use_container_width=True
    )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.subheader("ℹ️ Project Information")

info1, info2, info3 = st.columns(3)

with info1:

    st.info(
        """
        **Dataset**

        Breast Cancer Wisconsin

        569 observations

        30 numerical features
        """
    )


with info2:

    st.info(
        """
        **Classification**

        Binary classification

        0 = Malignant

        1 = Benign
        """
    )


with info3:

    st.info(
        """
        **Evaluation**

        Accuracy

        AUC

        Precision • Recall

        F1 • MCC
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Machine Learning Assignment 2 • Six-Model Classification Dashboard"
)
