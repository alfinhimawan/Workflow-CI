import os
import logging
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
import mlflow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TelcoChurnModelTrainer:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.model = None
        self.best_params = None

    def load_and_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        logger.info(f"Memuat data dari {self.data_path}")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"File tidak ditemukan di {self.data_path}")
            
        df = pd.read_csv(self.data_path)
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        logger.info("Melakukan data splitting (80% train, 20% test)...")
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_with_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        logger.info("Memulai Hyperparameter Tuning dengan GridSearchCV...")
        rf = RandomForestClassifier(random_state=42)
        
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        
        grid_search = GridSearchCV(
            estimator=rf, 
            param_grid=param_grid, 
            cv=3, 
            scoring='accuracy', 
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        logger.info(f"Tuning selesai. Parameter terbaik: {self.best_params}")

    def evaluate_and_log(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        if self.model is None:
            raise ValueError("Model belum dilatih. Jalankan train_with_tuning terlebih dahulu.")
            
        logger.info("Mengevaluasi model dan mencatat ke MLflow...")
        
        mlflow.set_experiment("Telco_Churn_CI")
        with mlflow.start_run():
            mlflow.log_params(self.best_params)
            
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)[:, 1]
            
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred)
            }
            
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            logger.info(f"Metrik evaluasi: {metrics}")
            
            mlflow.sklearn.log_model(self.model, "random_forest_model")
            self._generate_and_log_artifacts(y_test, y_pred, y_prob)
            logger.info("Semua metrik dan artefak berhasil dicatat di MLflow.")

    def _generate_and_log_artifacts(self, y_test: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> None:
        logger.info("Membuat grafik artefak tambahan...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        cm_path = os.path.join(base_dir, "confusion_matrix.png")
        plt.figure(figsize=(6,5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()
        
        roc_path = os.path.join(base_dir, "roc_curve.png")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path)
        plt.close()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'namadataset_preprocessing', 'dataset_clean.csv')
    
    trainer = TelcoChurnModelTrainer(data_path=data_path)
    
    try:
        X_train, X_test, y_train, y_test = trainer.load_and_split_data()
        trainer.train_with_tuning(X_train, y_train)
        trainer.evaluate_and_log(X_test, y_test)
        logger.info("Pipeline pelatihan selesai dengan sukses!")
    except Exception as e:
        logger.error(f"Pipeline gagal: {e}")

if __name__ == "__main__":
    main()
