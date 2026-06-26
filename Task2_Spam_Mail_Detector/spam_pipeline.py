import numpy as np
import pandas as pd
import urllib.request
import zipfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class SpamDetectorPipeline:
    def __init__(self):
        self.url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
        self.zip_path = "sms_spam.zip"
        self.extract_path = "sms_spam_extracted"
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        self.model = MultinomialNB()

    def download_and_prepare_data(self):
        print("📥 [SYSTEM] Downloading SMS Spam Collection Dataset...")
        urllib.request.urlretrieve(self.url, self.zip_path)
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.extract_path)
        
        data_file = os.path.join(self.extract_path, "SMSSpamCollection")
        self.df = pd.read_csv(data_file, sep='\t', names=['label', 'message'])
        self.df['label_num'] = self.df['label'].map({'ham': 0, 'spam': 1})
        
        X = self.df['message']
        y = self.df['label_num']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("💡 [SYSTEM] Dataset Loaded and Stratified Split Completed.")

    def plot_insights(self):
        print("📊 [SYSTEM] Generating Text Data Visualizations...")
        plt.figure(figsize=(6, 4))
        sns.countplot(data=self.df, x='label', palette='Set2')
        plt.title("Distribution of Spam vs Ham Messages")
        plt.xlabel("Message Type")
        plt.ylabel("Count")
        plt.show()

        self.df['message_len'] = self.df['message'].apply(len)
        plt.figure(figsize=(10, 5))
        sns.histplot(data=self.df, x='message_len', hue='label', bins=50, kde=True, palette='coolwarm')
        plt.title("Message Length Distribution")
        plt.xlabel("Length of Message")
        plt.ylabel("Frequency")
        plt.xlim(0, 300)
        plt.show()

    def train_pipeline(self):
        print("⚙️ [SYSTEM] Extracting TF-IDF Features & Training Naive Bayes Model...")
        X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        self.model.fit(X_train_tfidf, self.y_train)
        print("✅ [SUCCESS] Model Training Completed.")

    def evaluate_pipeline(self):
        X_test_tfidf = self.vectorizer.transform(self.X_test)
        y_pred = self.model.predict(X_test_tfidf)
        acc = accuracy_score(self.y_test, y_pred)
        
        print("\n" + "="*50)
        print(f"🔥 SPAM DETECTOR MODEL ACCURACY: {acc * 100:.2f}%")
        print("="*50)
        print("\n📋 CLASSIFICATION REPORT:\n", classification_report(self.y_test, y_pred, target_names=['Ham', 'Spam']))
        
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        plt.title("Spam Detector Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

if __name__ == "__main__":
    pipeline = SpamDetectorPipeline()
    pipeline.download_and_prepare_data()
    pipeline.plot_insights()
    pipeline.train_pipeline()
    pipeline.evaluate_pipeline()