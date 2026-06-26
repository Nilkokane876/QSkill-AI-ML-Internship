import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class HousePricePipeline:
    def __init__(self):
        self.housing_data = fetch_california_housing()
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.scaler = StandardScaler()
        self.model = LinearRegression()

    def prepare_data(self):
        print("📥 [SYSTEM] Fetching California Housing Dataset...")
        self.df = pd.DataFrame(data=self.housing_data.data, columns=self.housing_data.feature_names)
        self.df['Price'] = self.housing_data.target
        
        X = self.df[self.housing_data.feature_names]
        y = self.df['Price']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print("💡 [SYSTEM] Data Splitting and Feature Scaling (Normalization) Completed.")

    def plot_insights(self):
        print("📊 [SYSTEM] Generating Housing Data Visualizations...")
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Feature Correlation Matrix")
        plt.show()

        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=self.df, x='MedInc', y='Price', alpha=0.4, color='crimson')
        plt.title("Median Income vs House Price")
        plt.xlabel("Median Income")
        plt.ylabel("House Price (in $100,000s)")
        plt.show()

    def train_pipeline(self):
        print("⚙️ [SYSTEM] Training Linear Regression Model...")
        self.model.fit(self.X_train_scaled, self.y_train)
        print("✅ [SUCCESS] Model Training Completed.")

    def evaluate_pipeline(self):
        y_pred = self.model.predict(self.X_test_scaled)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        print("\n" + "="*50)
        print(f"🔥 HOUSE PRICE MODEL EVALUATION")
        print("="*50)
        print(f"📉 Mean Squared Error (MSE): {mse:.4f}")
        print(f"📈 R-squared (R2) Score: {r2 * 100:.2f}%")
        print("="*50)
        
        plt.figure(figsize=(8, 5))
        plt.scatter(self.y_test, y_pred, alpha=0.3, color='purple')
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=2)
        plt.title("Actual vs Predicted House Prices")
        plt.xlabel("Actual Prices")
        plt.ylabel("Predicted Prices")
        plt.show()

if __name__ == "__main__":
    pipeline = HousePricePipeline()
    pipeline.prepare_data()
    pipeline.plot_insights()
    pipeline.train_pipeline()
    pipeline.evaluate_pipeline()