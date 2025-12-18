import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run_improvement_experiment():
    print("=== TASK 4: MODEL IMPROVEMENT EXPERIMENT (Naive Bayes vs Logistic) ===")
    
    # Data mở rộng hơn chút để test
    texts = [
        "I love this movie", "This is terrible", "Great acting", "Waste of time", 
        "Highly recommended", "So boring", "Masterpiece", "I hate it",
        "Best film ever", "Worst movie ever", "I fell asleep", "Thrilling plot"
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 1. Baseline: Logistic Regression [cite: 238]
    baseline_model = LogisticRegression(solver='liblinear')
    baseline_model.fit(X_train_vec, y_train)
    base_pred = baseline_model.predict(X_test_vec)
    base_acc = accuracy_score(y_test, base_pred)

    # 2. Improvement: Naive Bayes 
    # Naive Bayes thường hoạt động rất tốt với văn bản
    improved_model = MultinomialNB()
    improved_model.fit(X_train_vec, y_train)
    imp_pred = improved_model.predict(X_test_vec)
    imp_acc = accuracy_score(y_test, imp_pred)

    print(f"Baseline (Logistic Regression) Accuracy: {base_acc:.4f}")
    print(f"Improved (Naive Bayes) Accuracy: {imp_acc:.4f}")
    
    if imp_acc > base_acc:
        print("-> Conclusion: Naive Bayes performed better on this small dataset.")
    elif imp_acc == base_acc:
        print("-> Conclusion: Both models performed equally.")
    else:
        print("-> Conclusion: Logistic Regression was actually better.")

if __name__ == "__main__":
    run_improvement_experiment()