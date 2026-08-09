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
    page_title="Breast Cancer ML Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_DATA_PATH = BASE_DIR / "train_data.csv"
TEST_DATA_PATH = BASE_DIR / "test_data.csv"


# ============================================================
# CONSTANTS
# ============================================================

RANDOM_STATE = 42

TARGET_COLUMN = "target"

CLASS_NAMES = {
    0: "Malignant",
    1: "Benign"
}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 1.8rem 2rem;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.25);
        background: linear-gradient(
            135deg,
            rgba(60,100,160,0.10),
            rgba(60,100,160,0.03)
        );
        margin-bottom: 1.4rem;
    }

    .hero-title {
        font-size: 2.35rem;
        font-weight: 750;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.72;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 0.6rem;
        margin-bottom: 0.7rem;
    }

    .info-card {
        border: 1px solid rgba(128,128,128,0.22);
        border-radius: 14px;
        padding: 1rem;
        min-height: 105px;
    }

    .winner-card {
        border: 1px solid rgba(60,150,80,0.35);
        border-radius: 14px;
        padding: 1.2rem;
        background: rgba(60,150,80,0.07);
    }

    .footer-note {
        text-align: center;
        opacity: 0.55;
        padding-top: 1.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🧬 Breast Cancer Classification ML Lab
        </div>

        <div class="hero-subtitle">
            Six-model classification benchmark with interactive
            evaluation, diagnostics and prediction analysis.
        </div>

    </div>
    """,
    unsafe_allow_html=True
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
        return False, "Missing values were detected in the feature columns."

    if df[TARGET_COLUMN].isnull().any():
        return False, "Missing values were detected in the target column."

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
# PREPARE TRAIN / TEST DATA
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

        "Logistic Regression": Pipeline([
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
        ]),


        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=3,
            random_state=RANDOM_STATE
        ),


        "kNN": Pipeline([
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
        ]),


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
# TRAINING FUNCTION
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
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    metrics = {

        "Accuracy":
            accuracy_score(
                y,
                predictions
            ),

        "AUC":
            roc_auc_score(
                y,
                probabilities
            ),

        "Precision":
            precision_score(
                y,
                predictions,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y,
                predictions,
                zero_division=0
            ),

        "F1 Score":
            f1_score(
                y,
                predictions,
                zero_division=0
            ),

        "MCC":
            matthews_corrcoef(
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
# STATUS
# ============================================================

st.success(
    f"Active model: **{model_name}**  |  "
    f"Training samples: **{len(X_train)}**  |  "
    f"Test samples: **{len(X_test)}**"
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">📊 Model Performance</div>',
    unsafe_allow_html=True
)

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
# SIX-MODEL BENCHMARK
# ============================================================

if compare_models:

    st.divider()

    st.markdown(
        '<div class="section-title">🏆 Six-Model Benchmark</div>',
        unsafe_allow_html=True
    )

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

    st.markdown(
        f"""
        <div class="winner-card">

        <b>🏆 Best Test-Set Model</b><br><br>

        <b>{winner["ML Model Name"]}</b><br>

        Accuracy: {winner["Accuracy"]:.4f}
        &nbsp; | &nbsp;
        AUC: {winner["AUC"]:.4f}
        &nbsp; | &nbsp;
        F1: {winner["F1"]:.4f}
        &nbsp; | &nbsp;
        MCC: {winner["MCC"]:.4f}

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">🔎 Model Diagnostics</div>',
    unsafe_allow_html=True
)

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

st.markdown(
    '<div class="section-title">🔬 Prediction Explorer</div>',
    unsafe_allow_html=True
)

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

    st.subheader(
        "🗃️ Test Data"
    )

    st.dataframe(
        test_df,
        use_container_width=True
    )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

info1, info2, info3 = st.columns(3)

with info1:

    st.markdown(
        """
        <div class="info-card">

        <b>Dataset</b><br>

        Breast Cancer Wisconsin<br>
        569 observations<br>
        30 numerical features

        </div>
        """,
        unsafe_allow_html=True
    )


with info2:

    st.markdown(
        """
        <div class="info-card">

        <b>Classification</b><br>

        Binary classification<br>
        0 = Malignant<br>
        1 = Benign

        </div>
        """,
        unsafe_allow_html=True
    )


with info3:

    st.markdown(
        """
        <div class="info-card">

        <b>Evaluation</b><br>

        Accuracy • AUC<br>
        Precision • Recall<br>
        F1 • MCC

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-note">
        Machine Learning Assignment 2 • Six-Model Classification Dashboard
    </div>
    """,
    unsafe_allow_html=True
)
