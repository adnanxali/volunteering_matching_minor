import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

class VolunteerRecommender:
    def __init__(self, model_path='./model/evaluation_results.json', graph_dir='./model/graphs'):
        self.model = None
        self.evaluation_results = {}
        self.model_path = model_path
        self.graph_dir = graph_dir
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(graph_dir, exist_ok=True)

    def generate_mock_data(self, n=1000):
        np.random.seed(42)
        skill_similarity = np.random.rand(n)
        location_proximity = np.random.rand(n)
        skill_overlap_count = np.random.randint(1, 6, n)
        matched = (skill_similarity + location_proximity + skill_overlap_count / 5 > 1.5).astype(int)

        return pd.DataFrame({
            'skill_similarity': skill_similarity,
            'location_proximity': location_proximity,
            'skill_overlap_count': skill_overlap_count,
            'matched': matched
        })

    def train_and_evaluate(self):
        df = self.generate_mock_data()
        X = df[['skill_similarity', 'location_proximity', 'skill_overlap_count']]
        y = df['matched']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        self.model = model

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_proba_train = model.predict_proba(X_train)[:, 1]
        y_proba_test = model.predict_proba(X_test)[:, 1]

        # Store evaluation results
        self.evaluation_results = {
            "regression_metrics": {
                "train": {
                    "rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
                    "mae": mean_absolute_error(y_train, y_pred_train),
                    "r2": r2_score(y_train, y_pred_train)
                },
                "test": {
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
                    "mae": mean_absolute_error(y_test, y_pred_test),
                    "r2": r2_score(y_test, y_pred_test)
                }
            },
            "classification_metrics": {
                "train": {
                    "accuracy": accuracy_score(y_train, y_pred_train),
                    "precision": precision_score(y_train, y_pred_train),
                    "recall": recall_score(y_train, y_pred_train),
                    "f1": f1_score(y_train, y_pred_train),
                    "auc": roc_auc_score(y_train, y_proba_train)
                },
                "test": {
                    "accuracy": accuracy_score(y_test, y_pred_test),
                    "precision": precision_score(y_test, y_pred_test),
                    "recall": recall_score(y_test, y_pred_test),
                    "f1": f1_score(y_test, y_pred_test),
                    "auc": roc_auc_score(y_test, y_proba_test)
                }
            },
            "roc_data": {},
            "feature_importance": []
        }

        # ROC data
        fpr, tpr, _ = roc_curve(y_test, y_proba_test)
        self.evaluation_results["roc_data"] = {
            "fpr": list(fpr),
            "tpr": list(tpr)
        }
        self._plot_roc_curve(fpr, tpr)

        # Feature importance
        feature_names = list(X.columns)
        importances = model.feature_importances_
        self.evaluation_results["feature_importance"] = [
            {"feature": name, "importance": round(score, 3)}
            for name, score in zip(feature_names, importances)
        ]
        self._plot_feature_importance(feature_names, importances)

        self.save_results()

        # NEW: Print evaluation metrics
        print("\n\U0001F4CA Evaluation Metrics (Train):")
        for metric, value in self.evaluation_results['classification_metrics']['train'].items():
            print(f"{metric.capitalize()}: {value:.4f}")

        print("\n\U0001F4CA Evaluation Metrics (Test):")
        for metric, value in self.evaluation_results['classification_metrics']['test'].items():
            print(f"{metric.capitalize()}: {value:.4f}")

        print("\n✅ Training complete. Results and graphs saved.")

    def _plot_roc_curve(self, fpr, tpr):
        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(self.graph_dir, 'roc_curve.png'))
        plt.close()

    def _plot_feature_importance(self, feature_names, importances):
        plt.figure()
        plt.barh(feature_names, importances, color='green')
        plt.xlabel('Importance')
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(self.graph_dir, 'feature_importance.png'))
        plt.close()

    def save_results(self):
        with open(self.model_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)

if __name__ == '__main__':
    recommender = VolunteerRecommender()
    recommender.train_and_evaluate()
