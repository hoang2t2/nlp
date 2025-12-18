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

# Báo cáo Lab 5 - Phần 3: Part-of-Speech Tagging với RNN

Phần 3 của Lab 5 nhằm áp dụng kiến trúc Mạng Nơ-ron Hồi quy (RNN) để giải quyết bài toán **Phân loại Token** (Token Classification), cụ thể là **Part-of-Speech (POS) Tagging** trên bộ dữ liệu UD-English. Bài toán này yêu cầu gán một nhãn loại từ cho **mỗi từ** trong câu.

## I. Các bước thực hiện (Implementation Steps)

Quá trình triển khai mô hình Phân loại Token dựa trên RNN (Recurrent Neural Network) được thực hiện qua 5 Task:

### A. Task 1: Tải và Tiền xử lý Dữ liệu

1.  **Đọc và Tách Dữ liệu:** Đã viết hàm `load_conllu` để đọc dữ liệu CoNLL-U .
    *   Loại bỏ các dòng metadata và các token dạng multiword .
    *   **Kết quả:** 12544 câu huấn luyện (Train) và 2001 câu phát triển (Dev) .

### B. Task 2: Tạo PyTorch Dataset và DataLoader

1.  **Xây dựng Vocabulary:** Đã tạo từ điển `word_to_ix` (19675 từ) và `tag_to_ix` (18 nhãn POS-tag) .
    *   Đã thêm các token đặc biệt (`<UNK>` và `<PAD>`) .
2.  **Xử lý Padding:** Đã viết hàm `collate_fn` tùy chỉnh cho `DataLoader`.
    *   Hàm này sử dụng `pad_sequence` để đệm (pad) các câu và nhãn theo độ dài lớn nhất trong batch .
    *   Đã trả về `lengths` cho `pack_padded_sequence`.

### C. Task 3: Xây dựng Mô hình RNN

*   **Kiến trúc:** Đã xây dựng mô hình Phân loại Token bao gồm 3 khối chính :
    1.  **`nn.Embedding`:** Chuyển đổi chỉ số từ thành vector 128 chiều .
    2.  **`nn.RNN`:** Lớp RNN (một chiều) với 128 chiều ẩn (`hidden_dim`) để xử lý chuỗi .
    3.  **`nn.Linear`:** Lớp tuyến tính ánh xạ output của RNN (128 chiều ẩn) sang 18 chiều (số lượng nhãn POS) [4].
*   **Nhận xét:** Mô hình đơn giản (không phải bidirectional) nhưng được nhận định là đủ hiệu quả .

### D. Task 4: Huấn luyện Mô hình

1.  **Cấu hình:** Sử dụng thuật toán tối ưu hóa **Adam** và Loss Function **`CrossEntropyLoss`**.
    *   **Xử lý Padding:** Thiết lập tham số **`ignore_index`** của `CrossEntropyLoss` bằng với index của nhãn padding để bỏ qua các token đệm khi tính toán tổn thất .
2.  **Quá trình Huấn luyện:** Huấn luyện trong 5 epochs .
3.  **Kết quả Loss:**
    *   Epoch 1/5 - Loss: 1.0720 
    *   Epoch 5/5 - Loss: 0.2714 
    *   Loss giảm đều, cho thấy mô hình học tốt .

### E. Task 5: Đánh giá Mô hình và Phân tích Kết quả

1.  **Chỉ số Đánh giá:** Chỉ tính độ chính xác (Accuracy) trên các token không phải là padding [7].
2.  **Kết quả Accuracy:**
    *   Train accuracy: 0.9307497910322274 .
    *   Dev accuracy: **0.8552626346972046** .

## II. Phân tích Kết quả

### A. Hiệu năng Định lượng

Mô hình RNN đơn giản đã đạt độ chính xác **85.53%** trên tập phát triển (Dev set) [6].

*   **Sức mạnh của RNN:** Kiến trúc RNN phù hợp cho Phân loại Token (POS Tagging) vì nó tạo ra một đầu ra dự đoán nhãn cho **mỗi từ**, đồng thời sử dụng trạng thái ẩn ($h_t$) để tích lũy ngữ cảnh từ các từ đứng trước .
*   **Ví dụ dự đoán câu mới:**
    *   Câu: "I love NLP" -> Dự đoán: `[('I', 'PRON'), ('love', 'VERB'), ('NLP', 'VERB')]` . (Lỗi ở "NLP" - VERB thay vì PROPN).
    *   Câu: "Students are studying in the library" -> Dự đoán: `('Students', 'VERB')` . (Lỗi nghiêm trọng, lẽ ra phải là NOUN).

### B. Phân tích Hạn chế (Lỗi Phân loại)

Lỗi phân loại (ví dụ: "Students" là VERB) cho thấy hạn chế của mô hình RNN đơn giản :
1.  **Thiếu Ngữ cảnh Tương lai:** RNN một chiều không thể sử dụng các từ đứng sau để xác định loại từ hiện tại. Trong nhiều trường hợp POS Tagging, thông tin từ tương lai là rất quan trọng .
2.  **Từ Hiếm:** Lỗi xảy ra do từ hiếm, trong trường hợp này, `Students` có thể bị nhầm lẫn với `[study, studies]` (liên quan đến VERB) .

## III. Khó khăn và Giải pháp

**Khó khăn 1: Xử lý dữ liệu tuần tự có độ dài biến đổi (Padding)**

*   *Vấn đề:* Các câu có độ dài khác nhau cần được đệm để xử lý trong cùng một batch .
*   *Giải pháp:* Sử dụng hàm **`collate_fn`** tùy chỉnh kết hợp với **`torch.nn.utils.rnn.pad_sequence`** .

**Khó khăn 2: Đảm bảo tính toán Loss và Accuracy chính xác**

*   *Vấn đề:* Các token đệm (`<PAD>`) không được tính vào Loss và Accuracy .
*   *Giải pháp:* Thiết lập tham số **`ignore_index`** của **`nn.CrossEntropyLoss`** bằng index của nhãn padding.

# Báo cáo Lab 5 - Phần 4: Named Entity Recognition (NER) với RNN

## I. Các bước thực hiện (Implementation Steps)

Chúng tôi đã xây dựng pipeline phân loại token cho bài toán NER, tuân thủ các bước chuẩn bị dữ liệu và mô hình RNN đơn giản.

### A. Task 1: Tải và Tiền xử lý Dữ liệu

1.  **Tải dữ liệu:** Đã tải bộ dữ liệu **CoNLL-2003** cho bài toán NER.
2.  **Tiền xử lý:** Dữ liệu được tiền xử lý theo chuẩn **IOB2** (Inside, Outside, Beginning) . Dữ liệu đã được tải bao gồm các trường word–POS–CHUNK–NER .
3.  **Xây dựng Vocabulary:** Đã tạo từ điển `word_to_ix` (kích thước 23626 từ) và `tag_to_ix` (kích thước 10 nhãn NER) .
    *   Đã thêm các token đặc biệt (`<UNK>`, `<PAD>`) để xử lý từ không nhìn thấy và padding.

### B. Task 2: Tạo PyTorch Dataset và DataLoader

1.  **Dataset:** Đã tạo lớp `NERDataset` để chuyển đổi câu và nhãn thành `word_ids` và `tag_ids`.
2.  **Xử lý Padding:** Đã viết hàm `collate_fn` tùy chỉnh để xử lý độ dài biến đổi của các chuỗi. Hàm này sử dụng `pad_sequence` để đệm (pad) các câu và nhãn theo độ dài lớn nhất trong batch .
3.  **Dữ liệu tuần tự:** Độ dài ban đầu của câu được lưu trữ (`lengths`) để có thể sử dụng `pack_padded_sequence` trong mô hình nhằm tránh tính toán trên các token padding .

### C. Task 3: Xây dựng Mô hình RNN

1.  **Kiến trúc:** Mô hình là một mạng RNN đơn giản được xây dựng từ các khối PyTorch (`nn.Module`):
    *   **`nn.Embedding`:** Chuyển đổi chỉ số từ (vocab size 23626) sang vector 128 chiều .
    *   **`nn.RNN`:** Lớp RNN xử lý chuỗi vector embedding. Kích thước trạng thái ẩn (`hidden_dim`) được đặt là 128.
    *   **`nn.Linear`:** Lớp cuối cùng ánh xạ output của RNN sang số lượng nhãn NER (10 nhãn) [8].
2.  **Tối ưu hóa:** Mô hình sử dụng kỹ thuật `pack/unpack` để đảm bảo rằng các phép tính toán được thực hiện chính xác và hiệu quả, tránh tính gradient trên các token đệm.
3.  **Nhận xét:** Mô hình RNN đơn giản được sử dụng, nhưng mô hình **Bi-LSTM** có thể là lựa chọn tối ưu hơn cho bài toán NER vì nó có khả năng nắm bắt ngữ cảnh từ cả hai phía (trước và sau) của từ.

### D. Task 4: Huấn luyện Mô hình

1.  **Cấu hình:** Sử dụng Loss Function `CrossEntropyLoss` .
    *   Quan trọng là thiết lập `ignore_index` của Loss Function bằng với giá trị padding của nhãn (`PAD_TAG`) để bỏ qua các vị trí đệm khi tính tổn thất .
2.  **Quá trình:** Mô hình được huấn luyện trong 3 epoch sử dụng thuật toán tối ưu hóa Adam (learning rate 0.001) .
3.  **Kết quả Loss:** Giá trị loss giảm đều qua các epoch , cho thấy mô hình đang học tốt:
    *   Epoch 1 - Loss: 0.1976
    *   Epoch 3 - Loss: 0.1000

## II. Hướng dẫn chạy code

Các thí nghiệm được thực hiện bằng PyTorch.

1.  **Cài đặt:** Đảm bảo thư viện PyTorch và `seqeval` (để tính metrics entity-level) đã được cài đặt .
2.  **Thực thi:** Chạy script chứa logic cho việc chuẩn bị dữ liệu, xây dựng mô hình và vòng lặp huấn luyện/đánh giá (ví dụ: `lab5_rnn_for_ner.py`) .

## III. Phân tích kết quả (Result Analysis)

Mô hình được đánh giá bằng Accuracy (token-level) và Precision–Recall–F1 (entity-level) sử dụng thư viện `seqeval` .

### A. Kết quả Tổng quan

| Chỉ số | Giá trị |
| :--- | :--- |
| **Accuracy (Token-level)** | 0.9129 |
| **F1-score (Micro, Entity-level)** | **0.5728** |
| Precision (Micro) | 0.5049 |
| Recall (Micro) | 0.6617 |

1.  **Độ chính xác Token-level (0.9129) cao** là điều dễ hiểu vì phần lớn nhãn NER là 'O' (Outside) [4].
2.  **F1-score Entity-level (0.5728)** là chỉ số quan trọng nhất cho NER. Kết quả này là mức khởi điểm hợp lý cho một mô hình RNN đơn giản (không phải Bi-LSTM) .

### B. Hiệu suất theo từng Thực thể

| Loại thực thể | Precision | Recall | F1-score |
| :--- | :--- | :--- | :--- |
| LOC (Địa điểm) | 0.42 | 0.80 | 0.55 |
| MISC (Khác) | 0.50 | 0.63 | 0.56 |
| ORG (Tổ chức) | 0.47 | 0.58 | 0.52 |
| PER (Người) | **0.73** | 0.60 | **0.66** |

*   **Thực thể PER (Tên người)** có F1 cao nhất (0.66) và Precision cao nhất (0.73) [14].
*   **Thực thể LOC (Địa điểm)** có Recall cao nhất (0.80), cho thấy mô hình tìm được phần lớn các thực thể địa điểm, nhưng Precision thấp (0.42) chỉ ra rằng nó cũng dự đoán sai nhiều thực thể khác là LOC .

### C. Ví dụ Dự đoán Câu Mới

*   **Câu:** “VNU University is located in Hanoi”
*   **Dự đoán:** `[('VNU','B-ORG'), ('University','I-ORG'), ('is','O'), ('located','O'), ('in','O'), ('Hanoi','B-LOC')]`.

**Nhận xét:** Dự đoán cho thấy mô hình RNN đã xác định và gán nhãn thực thể chính xác cho hai thực thể phức tạp: **"VNU University"** (B-ORG, I-ORG) và **"Hanoi"** (B-LOC) .

## IV. Nêu khó khăn và giải pháp (Challenges and Solutions)

**Khó khăn 1: Xử lý độ dài chuỗi biến đổi và Masking:**

*   *Vấn đề:* Token Classification yêu cầu tính toán đầu ra cho mỗi token, nhưng độ dài câu khác nhau cần phải đệm (pad) .
*   *Giải pháp:* Sử dụng `collate_fn` tùy chỉnh và `torch.nn.utils.rnn.pad_sequence` để đệm. Sau đó, sử dụng kỹ thuật `pack/unpack` hoặc lưu trữ `lengths` để đảm bảo RNN chỉ tính toán trên dữ liệu thực tế .

**Khó khăn 2: Đảm bảo Loss Function bỏ qua Padding:**

*   *Vấn đề:* Các token đệm trong nhãn (`PAD_TAG`) không được tính vào giá trị loss và accuracy.
*   *Giải pháp:* Thiết lập tham số **`ignore_index`** trong `nn.CrossEntropyLoss` bằng với chỉ số của nhãn padding. Đồng thời, chỉ tính Accuracy trên các token có nhãn thật .

**Khó khăn 3: Hạn chế của Mô hình RNN đơn giản:**

*   *Vấn đề:* Mô hình RNN đơn giản không thể xử lý tốt các phụ thuộc xa (long-range dependencies) và không nắm bắt được ngữ cảnh từ tương lai (các từ đứng sau).
*   *Giải pháp (Đề xuất):* Chuyển sang sử dụng kiến trúc **Bi-LSTM** (Bidirectional LSTM). Bi-LSTM sử dụng hai lớp LSTM chạy ngược và xuôi, cho phép mô hình nắm bắt ngữ cảnh từ cả hai phía, điều này cực kỳ quan trọng cho các tác vụ phân loại token như NER và POS Tagging.
