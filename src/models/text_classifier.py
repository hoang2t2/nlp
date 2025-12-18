from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List, Dict

class TextClassifier:
    def __init__(self, vectorizer):
        """
        Khởi tạo với một Vectorizer (Tfidf hoặc Count)
        [cite: 257]
        """
        self.vectorizer = vectorizer
        self.model = None

    def fit(self, texts: List[str], labels: List[int]):
        """
        Huấn luyện model
        [cite: 259-263]
        """
        # 1. Vectorize text
        X = self.vectorizer.fit_transform(texts)
        
        # 2. Initialize LogisticRegression (solver='liblinear' tốt cho data nhỏ)
        self.model = LogisticRegression(solver='liblinear')
        
        # 3. Train
        self.model.fit(X, labels)
        print("Model training completed.")

    def predict(self, texts: List[str]) -> List[int]:
        """
        Dự đoán nhãn cho text mới
        [cite: 264-267]
        """
        if not self.model:
            raise Exception("Model has not been trained yet!")
            
        X = self.vectorizer.transform(texts)
        return self.model.predict(X).tolist()

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Tính toán các chỉ số đánh giá
        [cite: 268-270]
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='binary'),
            "recall": recall_score(y_true, y_pred, average='binary'),
            "f1_score": f1_score(y_true, y_pred, average='binary')
        }