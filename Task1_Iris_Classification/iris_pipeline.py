import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class IrisPipeline:
    def __init__(self):
        self.iris = load_iris()
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.best_model = None

    def prepare_data(self):
        self.df = pd.DataFrame(data=self.iris.data, columns=self.iris.feature_names)
        self.df['target'] = self.iris.target
        self.df['species'] = self.df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        X = self.df[self.iris.feature_names]
        y = self.df['target']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("💡 [SYSTEM] Data Pipeline Ready. Stratified Splitting Completed.")

    def plot_insights(self):
        print("📊 [SYSTEM] Generating Advanced Visualizations...")
        sns.pairplot(self.df, hue='species', palette='Dark2', markers=["o", "s", "D"])
        plt.suptitle("Advanced Feature Interaction - Iris Dataset", y=1.02)
        plt.show()

        plt.figure(figsize=(10, 5))
        sns.violinplot(x='species', y='petal length (cm)', data=self.df, palette='Pastel1')
        plt.title("Petal Length Density Distribution Across Species")
        plt.show()

    def train_with_hypertuning(self):
        print("⚙️ [SYSTEM] Optimizing Random Forest Model using GridSearchCV...")
        rf = RandomForestClassifier(random_state=42)
        
        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [3, 5, None],
            'criterion': ['gini', 'entropy']
        }
        
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        
        self.best_model = grid_search.best_estimator_
        print(f"✅ [SUCCESS] Best Parameters Found: {grid_search.best_params_}")

    def evaluate_pipeline(self):
        y_pred = self.best_model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        
        print("\n" + "="*50)
        print(f"🔥 INDUSTRIAL PIPELINE ACCURACY: {acc * 100:.2f}%")
        print("="*50)
        print("\n📋 CLASSIFICATION REPORT:\n", classification_report(self.y_test, y_pred, target_names=self.iris.target_names))
        
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='magma', xticklabels=self.iris.target_names, yticklabels=self.iris.target_names)
        plt.title("Production Model Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

if __name__ == "__main__":
    pipeline = IrisPipeline()
    pipeline.prepare_data()
    pipeline.plot_insights()
    pipeline.train_with_hypertuning()
    pipeline.evaluate_pipeline()
