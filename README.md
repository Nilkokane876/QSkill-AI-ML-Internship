# QSkill-AI-ML-Internship
# 🤖 QSkill AI/ML Internship — Task Submissions

**Intern:** Neel | **Batch:** QSkill AIML Internship  
**Tech Stack:** Python · scikit-learn · pandas · seaborn · matplotlib

---

## 📁 Project Structure

```
QSkill_AIML_Internship_3/
├── Task1_Iris_Classification/
│   ├── iris_pipeline.py
│   ├── iris_data_visualization.png
│   ├── model_training_accuracy.png
│   └── iris_prediction_output.png
│
├── Task2_Spam_Mail_Detector/
│   ├── spam_pipeline.py
│   ├── dataset_overview.png
│   ├── spam_model_evaluation.png
│   └── spam_prediction_test.png
│
└── Task3_House_Price_Prediction/
    ├── house_price_pipeline.py
    ├── house_data_analysis.png
    └── regression_model_metrics.png
```

---

## ✅ Task 1 — Iris Species Classification

### 🎯 Objective
Classify iris flowers into 3 species — *Setosa*, *Versicolor*, and *Virginica* — using a Random Forest Classifier with hyperparameter tuning.

### 📦 Dataset
- **Source:** `sklearn.datasets.load_iris` (built-in)
- **Features:** Sepal length, sepal width, petal length, petal width
- **Classes:** 3 species | **Samples:** 150

### ⚙️ Approach
- **Model:** `RandomForestClassifier`
- **Tuning:** `GridSearchCV` with 5-fold cross-validation
- **Hyperparameters searched:** `n_estimators`, `max_depth`, `criterion`
- **Split:** 80% train / 20% test (stratified)

### 📊 Visualizations
- Pairplot of feature interactions (colored by species)
- Violin plot of petal length distribution
- Confusion matrix heatmap

### 🏆 Result
- Achieved high classification accuracy via GridSearch optimization
- Best parameters auto-selected by cross-validation

### ▶️ How to Run
```bash
cd Task1_Iris_Classification
pip install numpy pandas matplotlib seaborn scikit-learn
python iris_pipeline.py
```

---

## ✅ Task 2 — Spam Mail Detector

### 🎯 Objective
Build an SMS/Email spam detector using NLP (TF-IDF) and a Naive Bayes classifier.

### 📦 Dataset
- **Source:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (auto-downloaded)
- **Size:** ~5,574 labeled messages
- **Labels:** `ham` (legitimate) / `spam`

### ⚙️ Approach
- **Preprocessing:** Lowercasing, English stopword removal via `TfidfVectorizer`
- **Model:** `MultinomialNB` (Naive Bayes — ideal for text classification)
- **Split:** 80% train / 20% test (stratified)

### 📊 Visualizations
- Bar chart: Spam vs Ham distribution
- Histogram: Message length distribution by label (with KDE)
- Confusion matrix heatmap

### 🏆 Result
- High accuracy spam detection leveraging TF-IDF feature extraction
- Naive Bayes performs exceptionally well on sparse text data

### ▶️ How to Run
```bash
cd Task2_Spam_Mail_Detector
pip install numpy pandas matplotlib seaborn scikit-learn
python spam_pipeline.py
```
> **Note:** The script auto-downloads the dataset on first run. Internet connection required.

---

## ✅ Task 3 — House Price Prediction

### 🎯 Objective
Predict California housing prices using Linear Regression with feature scaling.

### 📦 Dataset
- **Source:** `sklearn.datasets.fetch_california_housing` (built-in)
- **Features:** 8 (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude)
- **Target:** Median house value (in $100,000s)

### ⚙️ Approach
- **Model:** `LinearRegression`
- **Scaling:** `StandardScaler` (normalization before training)
- **Split:** 80% train / 20% test

### 📊 Visualizations
- Correlation heatmap across all features
- Scatter plot: Median Income vs House Price
- Actual vs Predicted house prices plot

### 🏆 Metrics
| Metric | Description |
|--------|-------------|
| MSE (Mean Squared Error) | Measures average squared prediction error |
| R² Score | Proportion of variance explained by the model |

### ▶️ How to Run
```bash
cd Task3_House_Price_Prediction
pip install numpy pandas matplotlib seaborn scikit-learn
python house_price_pipeline.py
```

---

## 🛠️ Global Requirements

Install all dependencies at once:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

**Python version:** 3.8+

---

## 🧠 Key Learnings

- End-to-end ML pipeline design using OOP (class-based architecture)
- Supervised learning: Classification & Regression
- NLP basics: TF-IDF vectorization for text data
- Model evaluation: Accuracy, Classification Report, MSE, R²
- Hyperparameter tuning with GridSearchCV
- Data visualization with seaborn & matplotlib

---

*Submitted as part of QSkill AI/ML Internship Program*
