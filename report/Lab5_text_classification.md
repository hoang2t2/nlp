# Báo cáo Lab 5: Text Classification (Phân loại văn bản)

Mục tiêu của Lab 5 là xây dựng một pipeline phân loại văn bản hoàn chỉnh, sử dụng mô hình học máy giám sát (supervised learning) và đánh giá các kỹ thuật biểu diễn văn bản khác nhau (TF-IDF, Word2Vec, Embeddings).

## I. Các bước thực hiện (Implementation Steps)

Em đã hoàn thành các Task sau đây để xây dựng và đánh giá hệ thống phân loại văn bản:

### A. Task 1 & 2: Triển khai Mô hình Baseline (Scikit-learn)

1.  **TextClassifier Implementation:** Đã triển khai lớp `TextClassifier` trong `src/models/text_classifier.py`.
    *   **Constructor:** Chấp nhận một instance `Vectorizer` (ví dụ: `TfidfVectorizer`) .
    *   **`fit` method:** Sử dụng `vectorizer` để `fit_transform` văn bản huấn luyện thành ma trận đặc trưng $X$, sau đó khởi tạo và huấn luyện mô hình **LogisticRegression** của scikit-learn (sử dụng `solver='liblinear'` cho dataset nhỏ) .
    *   **`predict` method:** Sử dụng `vectorizer` để `transform` văn bản mới thành ma trận $X$ và dùng mô hình đã huấn luyện để dự đoán .
    *   **`evaluate` method:** Tính toán các chỉ số đánh giá quan trọng: **Accuracy, Precision, Recall, và F1-score** bằng cách sử dụng các hàm từ `sklearn.metrics`.
2.  **Basic Test Case:** Đã tạo file `test/lab5_test.py`. Test case này sử dụng dataset mẫu (`texts`, `labels`) , chia dữ liệu bằng `train_test_split`, huấn luyện classifier và in ra các metrics .

### B. Task 3: Phân tích và chạy Ví dụ Spark ML

*   Đã chạy thành công script `test/lab5_spark_sentiment_analysis.py`.
*   **Spark ML Pipeline:** Script này minh họa việc sử dụng PySpark để xây dựng pipeline phân loại văn bản cho dữ liệu lớn (sử dụng `data/sentiments.csv`) [10, 11]. Pipeline bao gồm các thành phần sau:
    1.  **Tokenizer:** Tách văn bản thành các token .
    2.  **StopWordsRemover:** Loại bỏ các stop words phổ biến .
    3.  **HashingTF:** Chuyển đổi tokens thành vector đặc trưng có kích thước cố định bằng kỹ thuật băm .
    4.  **IDF (Inverse Document Frequency):** Điều chỉnh trọng số vector, giảm trọng số của các thuật ngữ xuất hiện thường xuyên.
    5.  **LogisticRegression (lr):** Mô hình học máy được huấn luyện trên các vector TF-IDF đã được chuẩn hóa .

### C. Task 4: Thí nghiệm Cải tiến Mô hình (Model Improvement)

 Đã thực hiện thử nghiệm cải tiến bằng cách sử dụng **Advanced Embedding Methods** và **Neural Networks** (LSTM), một trong các phương pháp được đề xuất.

*   **Kỹ thuật:** Thay vì sử dụng các mô hình tuyến tính trên vector thưa (TF-IDF), chúng tôi đã thử nghiệm các phương pháp biểu diễn ngữ nghĩa dày đặc kết hợp với mạng nơ-ron:
    *   Word2Vec (Avg) + Dense .
    *   Pre-trained Embedding (GloVe/FastText) + LSTM .
    *   Embedding huấn luyện từ đầu (Scratch Embedding) + LSTM .
*   **Thực thi:** Tạo môi trường thử nghiệm nâng cao để so sánh hiệu suất của các pipeline này với Baseline .

## II. Hướng dẫn chạy code (Code Execution Guide)

1.  **Cài đặt:** Đảm bảo các thư viện cần thiết (`scikit-learn`, `pyspark`, và thư viện Deep Learning nếu có) đã được cài đặt .
2.  **Chạy Baseline:** Để kiểm tra mô hình TF-IDF + Logistic Regression, chạy `python test/lab5_test.py` .
3.  **Chạy Spark:** Để kiểm tra pipeline xử lý dữ liệu lớn, chạy `python test/lab5_spark_sentiment_analysis.py` .
4.  **Chạy Thí nghiệm Cải tiến:** Chạy script thử nghiệm nâng cao (ví dụ: `test/lab5_improvement_test.py`) để tái tạo các kết quả Embedding và LSTM.

## III. Phân tích kết quả (Result Analysis)

Dưới đây là các chỉ số hiệu suất của các pipeline được thử nghiệm, bao gồm mô hình Baseline và các mô hình cải tiến.

| Pipeline | Macro F1 (Test) | Test Accuracy | Test Loss |
| :--- | :--- | :--- | :--- |
| **Baseline (TF-IDF + Logistic Regression)** | **0.84** | **0.84** | N/A |
| Word2Vec (Avg) + Dense | 0.81 | 0.82 | 0.67 |
| Pre-trained Embedding + LSTM | 0.84 | 0.84 | 0.56 |
| **Cải tiến (Scratch Embedding + LSTM)** | **0.85** | **0.85** | **0.61** |

### A. Báo cáo Hiệu suất Mô hình Baseline

Mô hình Baseline sử dụng **TF-IDF và Logistic Regression** đạt **Accuracy 0.84** và **Macro F1 0.84**.

*   Mô hình Logistic Regression hoạt động hiệu quả như một mô hình tuyến tính đơn giản, cho thấy các đặc trưng TF-IDF đã phân tách tốt giữa các lớp (positive/negative).

### B. Báo cáo Hiệu suất Mô hình Cải tiến

Mô hình tốt nhất trong các thử nghiệm cải tiến là **Scratch Embedding + LSTM**, đạt **Accuracy 0.85** và **Macro F1 0.85**.

### C. So sánh và Phân tích Hiệu quả Cải tiến

1.  **Mô hình tuyến tính vs. Mạng nơ-ron:** Mô hình LSTM sử dụng Embedding huấn luyện từ đầu đạt hiệu suất cao nhất (0.85 F1), tuy nhiên, mức cải thiện so với mô hình Baseline (0.84 F1) là rất nhỏ. Điều này cho thấy rằng đối với dataset này, phần lớn thông tin phân loại đã được capture bởi các đặc trưng tần suất đơn giản (TF-IDF).
2.  **Hiệu suất Word2Vec (Avg):** Phương pháp Word2Vec trung bình cho kết quả kém hơn (0.81 F1), có thể do việc lấy trung bình vector làm mất đi thông tin về thứ tự từ và cấu trúc ngữ pháp quan trọng cho phân loại .
3.  **Pre-trained vs. Scratch Embedding:** Pre-trained Embedding + LSTM cho kết quả bằng với Baseline (0.84 F1), trong khi Embedding huấn luyện từ đầu (Scratch) hoạt động tốt hơn một chút (0.85 F1). Điều này gợi ý rằng Embedding được học trực tiếp từ dataset có thể đã nắm bắt được các sắc thái từ vựng cụ thể của miền dữ liệu hiệu quả hơn các vector GloVe/FastText tổng quát.

## IV. Nêu khó khăn và giải pháp (Challenges and Solutions)

**Khó khăn 1: Kết nối Vectorizer và TextClassifier (Task 1):**
*   *Vấn đề:* Đảm bảo `TextClassifier` sử dụng đúng `Vectorizer` trong cả `fit` (`fit_transform`) và `predict` (`transform`) để tránh rò rỉ dữ liệu (Data Leakage) [7].
*   *Giải pháp:* Sử dụng Dependency Injection, truyền instance `Vectorizer` vào constructor của `TextClassifier`. Đảm bảo chỉ gọi `transform` trên dữ liệu kiểm tra [6].

**Khó khăn 2: Phân tích Pipeline Spark ML (Task 3):**
*   *Vấn đề:* Hiểu rõ vai trò của từng thành phần trong Spark ML Pipeline, đặc biệt là sự khác biệt giữa `HashingTF` và `IDF`.
*   *Giải pháp:* Phân tích sâu mã nguồn `lab5_spark_sentiment_analysis.py`, nhận ra `HashingTF` chuyển đổi token thành vector băm cố định, và `IDF` điều chỉnh trọng số để giảm giá trị của các từ phổ biến (giống như TF-IDF truyền thống).

**Khó khăn 3: Đánh giá Mô hình (Task 1):**
*   *Vấn đề:* Cần tính toán đồng thời nhiều chỉ số đánh giá (Accuracy, Precision, Recall, F1) .
*   *Giải pháp:* Sử dụng hàm `evaluate` của `TextClassifier` để tính toán và trả về các chỉ số này dưới dạng dictionary, tận dụng các hàm metrics của `sklearn`.
