# Báo cáo Lab 5 - Phần 2: Phân loại Văn bản với Mạng Nơ-ron Hồi quy (RNN/LSTM)

Báo cáo này tập trung vào việc triển khai và phân tích 4 pipeline phân loại văn bản (Task 1 đến Task 4) theo yêu cầu của **Part 2: RNNs cho Phân loại Văn bản** , và tuân thủ các tiêu chí phân tích, báo cáo kết quả theo **Task 5**.

## I. Các bước thực hiện (Implementation Steps)

### A. Task 1: Baseline Model 1 (TF-IDF + Logistic Regression)

*   **Mục tiêu:** Thiết lập mô hình tuyến tính cổ điển, sử dụng mô hình Bag-of-Words .
*   **Triển khai:** Sử dụng `TfidfVectorizer` để chuyển đổi văn bản thành vector tần suất trọng số và `LogisticRegression` để huấn luyện . Mô hình này bỏ qua thứ tự và ngữ cảnh của từ .

### B. Task 2: Baseline Model 2 (Word2Vec + Dense Layer)

*   **Mục tiêu:** Sử dụng vector dày đặc (dense vector) nhưng vẫn bỏ qua thứ tự từ .
*   **Triển khai:** Huấn luyện mô hình **Word2Vec** trên dữ liệu văn bản. Vector biểu diễn câu được tạo ra bằng cách tính **trung bình cộng (average)** của các word embedding trong câu [3, 7]. Sau đó, vector trung bình này được đưa vào một mạng nơ-ron truyền thẳng (`Dense` Layer) để phân loại .

### C. Task 3: LSTM Model with Pre-trained Embeddings

*   **Mục tiêu:** Sử dụng mô hình chuỗi (Sequence Model) để nắm bắt thứ tự và ngữ cảnh .
*   **Triển khai:**
    1.  **Tiền xử lý:** Chuyển văn bản thành chuỗi chỉ số (`Tokenizer`) và đệm (`pad_sequences`) về cùng độ dài [9, 10].
    2.  **Embedding Layer:** Khởi tạo bằng ma trận embedding đã được huấn luyện trước (Pre-trained Word2Vec/GloVe) và được **đóng băng (`trainable=False`)** .
    3.  **Mô hình:** Kiến trúc **LSTM** (Long Short-Term Memory) được sử dụng để xử lý chuỗi vector .

### D. Task 4: LSTM Model with End-to-End Training

*   **Mục tiêu:** Đánh giá hiệu suất của vector nhúng được học chuyên biệt (End-to-End Training) .
*   **Triển khai:** Kiến trúc tương tự Task 3, nhưng lớp **Embedding** được khởi tạo ngẫu nhiên và **được huấn luyện (`trainable=True`)** cùng với các trọng số của LSTM .

## II. Hướng dẫn chạy code (Code Execution Guide)

1.  **Cài đặt:** Đảm bảo `scikit-learn`, `gensim`, và `tensorflow/keras` (hoặc PyTorch) đã được cài đặt. Các bài tập lý thuyết PyTorch nền tảng đã được hoàn thành .
2.  **Tải dữ liệu:** Giải nén bộ dữ liệu HWU .
3.  **Thực thi:** Chạy lần lượt các script tương ứng với 4 Nhiệm vụ (ví dụ: `lab5_rnns_text_classification.py`) để huấn luyện và đánh giá trên tập test, ghi nhận kết quả F1-macro và Test Loss .

## III. Phân tích kết quả (Result Analysis)

Dưới đây là bảng tổng hợp kết quả của 4 pipeline trên tập kiểm tra (Test Set) :

| Model | F1-macro | Test Loss |
| :--- | :--- | :--- |
| **TF-IDF + Logistic Regression (Baseline 1)** | **0.822567** | 1.052858 |
| **Word2Vec Avg + Dense (Baseline 2)** | 0.796896 | **0.724759** |
| LSTM + Pretrained Embedding (Task 3) | 0.640867 | 1.050412 |
| LSTM + Scratch Embedding (Task 4) | 0.000533 | 4.128992 |

### A. So sánh Hiệu năng của 4 Mô hình (Task 5)

1.  **Mô hình Tốt nhất (Baseline 1):** **TF-IDF + Logistic Regression** đạt F1-macro cao nhất (0.822567). Điều này chỉ ra rằng, đối với bộ dữ liệu phân loại ý định này, các từ khóa quan trọng và tần suất của chúng (đặc trưng TF-IDF) là yếu tố quyết định, và mô hình tuyến tính là đủ để phân tách các lớp .
2.  **Mô hình Kém nhất (Task 4):** **LSTM + Scratch Embedding** cho F1-macro cực kỳ thấp (0.000533) và Loss rất cao (4.128992). Đây là dấu hiệu của việc mô hình **thất bại hoàn toàn trong việc hội tụ** .

### B. Phân tích sức mạnh và hạn chế của mô hình chuỗi (LSTM)

Mặc dù lý thuyết chỉ ra rằng LSTM có khả năng nắm bắt ngữ cảnh và giải quyết vấn đề phụ thuộc xa (nhờ các Cổng Quên, Đầu vào, và Ô nhớ) [8, 16, 17], kết quả thực nghiệm lại cho thấy hiệu suất của LSTM bị suy giảm nghiêm trọng:

*   **Word2Vec Avg vs. LSTM Pretrained:** Mô hình Word2Vec trung bình (Baseline 2, F1: 0.796) hoạt động tốt hơn đáng kể so với LSTM Pretrained (Task 3, F1: 0.640). Điều này củng cố rằng việc lấy trung bình vector chỉ làm mất thông tin thứ tự , nhưng mô hình LSTM bị đóng băng trọng số đã không thể tối ưu hóa và học các phụ thuộc ngữ cảnh hiệu quả.
*   **Thất bại Huấn luyện (Task 4):** Việc mô hình LSTM Scratch không hội tụ là do **Exploding Gradient (Đạo hàm bùng nổ)** [18, 19]. Gradient bùng nổ xảy ra khi gradient được nhân lặp lại qua nhiều bước thời gian (BPTT), làm cho trọng số thay đổi đột ngột [18, 20], khiến mô hình không thể học được.

**Kết luận Phân tích:** Các mô hình LSTM đã thất bại trong việc chứng minh sức mạnh của mô hình chuỗi trong thí nghiệm này, không phải do lý thuyết mà do các vấn đề tối ưu hóa nghiêm trọng khi huấn luyện mô hình phức tạp (LSTM) từ đầu .

## IV. Nêu khó khăn và giải pháp (Challenges and Solutions)

**Khó khăn 1: Thất bại trong hội tụ của LSTM (Task 4):**

*   *Vấn đề:* Mô hình LSTM + Scratch không hội tụ (F1 ~ 0) do khả năng cao là Exploding Gradient .
*   *Giải pháp (Được đề xuất):* Áp dụng **Gradient Clipping** (cắt bớt giá trị gradient khi chúng quá lớn) và điều chỉnh tốc độ học (learning rate) trong quá trình huấn luyện để đảm bảo mô hình hội tụ ổn định.

**Khó khăn 2: Xử lý dữ liệu tuần tự và Padding:**

*   *Vấn đề:* Các câu có độ dài khác nhau cần được chuẩn hóa để đưa vào lớp Embedding/LSTM [10, 21].
*   *Giải pháp:* Sử dụng `pad_sequences` của Keras/TensorFlow để đệm các chuỗi chỉ số về cùng một độ dài tối đa (`max_len`) .

**Khó khăn 3: Đảm bảo tính nhất quán của Vocab:**

*   *Vấn đề:* Đảm bảo `vocab_size` và `max_len` được áp dụng đồng nhất cho các mô hình 3 và 4 .
*   *Giải pháp:* Sử dụng cùng một instance `Tokenizer` (`tokenizer.word_index`) và cùng một tham số `max_len` cho tất cả các tập dữ liệu (train/val/test) và các mô hình LSTM.
