"""
modelling.py - Untuk MLProject (Kriteria 3)
Versi modelling yang kompatibel dengan MLflow Project.
"""

import pandas as pd
import mlflow
import mlflow.sklearn
import os
import logging
import joblib
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(data_dir="iris_preprocessing"):
    """Load preprocessed data."""
    train_df = pd.read_csv(os.path.join(data_dir, "iris_train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "iris_test.csv"))

    feature_cols = [c for c in train_df.columns if c != "species"]
    X_train = train_df[feature_cols].values
    y_train = train_df["species"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["species"].values

    return X_train, y_train, X_test, y_test, feature_cols


def plot_confusion_matrix_fn(y_true, y_pred, save_path="confusion_matrix.png"):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    class_names = ["setosa", "versicolor", "virginica"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def main():
    # Baca parameter dari environment variables (set oleh MLflow Project)
    n_estimators = int(os.environ.get("N_ESTIMATORS", 100))
    max_depth_val = os.environ.get("MAX_DEPTH", "None")
    max_depth = None if max_depth_val == "None" else int(max_depth_val)
    random_state = int(os.environ.get("RANDOM_STATE", 42))

    logger.info(
        f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}, random_state={random_state}"  # noqa: E501
    )

    # Set MLflow tracking URI
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"
    )  # noqa: E501
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Iris CI Pipeline - Alfin Himawan Santosa")

    logger.info("Loading data...")
    X_train, y_train, X_test, y_test, feature_cols = load_data()

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("model_type", "RandomForestClassifier")

        # Train model
        logger.info("Training model...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted", recall)
        mlflow.log_metric("f1_score_weighted", f1)

        logger.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        # Log confusion matrix plot
        cm_path = plot_confusion_matrix_fn(y_test, y_pred)
        mlflow.log_artifact(cm_path, "plots")

        # Log classification report
        report = classification_report(
            y_test, y_pred, target_names=["setosa", "versicolor", "virginica"]
        )
        report_path = "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path, "reports")

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Simpan model lokal juga
        os.makedirs("model_output", exist_ok=True)
        joblib.dump(model, "model_output/model.pkl")
        mlflow.log_artifact("model_output/model.pkl", "model_output")

        logger.info(f"Training selesai. Accuracy: {accuracy:.4f}")

        # Print run ID untuk digunakan di workflow
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow Run ID: {run_id}")

        # Simpan run ID ke file
        with open("mlflow_run_id.txt", "w") as f:
            f.write(run_id)


if __name__ == "__main__":
    main()
