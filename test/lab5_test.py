import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from src.models.text_classifier import TextClassifier

def run_basic_test():
    print("=== TASK 2: BASIC TEST CASE ===")
    
    # 1. Dataset mẫu [cite: 242-250]
    texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommend this, a masterpiece.",
        "Could not finish watching, so bad."
    ]
    labels = [1, 0, 1, 0, 1, 0] # 1: Positive, 0: Negative

    # 2. Split Data [cite: 274]
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

    # 3. Setup Pipeline [cite: 276-277]
    # Bạn có thể dùng RegexTokenizer từ Lab 1 nếu muốn, ở đây dùng sklearn cho chuẩn
    vectorizer = TfidfVectorizer()
    classifier = TextClassifier(vectorizer)

    # 4. Train [cite: 278]
    print(f"Training on {len(X_train)} samples...")
    classifier.fit(X_train, y_train)

    # 5. Predict [cite: 279]
    print(f"Testing on {len(X_test)} samples...")
    y_pred = classifier.predict(X_test)
    print(f"True Labels: {y_test}")
    print(f"Predictions: {y_pred}")

    # 6. Evaluate [cite: 280]
    metrics = classifier.evaluate(y_test, y_pred)
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    run_basic_test()