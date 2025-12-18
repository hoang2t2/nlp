## PHẦN II: LAB 4 (WORD EMBEDDINGS)

### A.Các bước thực hiện 

Lab này tập trung vào việc triển khai mô hình nhúng từ (Embedding), nhằm nắm bắt ngữ nghĩa của từ .

#### 1. Task 1 & 2: Tải model có sẵn và Nhúng từ/câu (Gensim)
*   **Tải Model Pre-trained:** Đã triển khai lớp `WordEmbedder` (`src/representations/word_embedder.py`) và tải thành công mô hình **GloVe pre-trained** (`glove-wiki-gigaword-50` - vector 50 chiều) từ Gensim .
*   **Khám phá Embedding:** Đã triển khai `get_vector` (có xử lý OOV), `get_similarity` (tính độ tương đồng cosine), và `get_most_similar` (tìm từ đồng nghĩa).
*   **Document Embedding (Task 3):** Đã triển khai `embed_document` để tính vector nhúng tài liệu bằng cách lấy **trung bình cộng (element-wise mean)** của tất cả các vector từ trong văn bản, bỏ qua các từ OOV.

#### 2.Task: Huấn luyện model trên tập dữ liệu nhỏ (Gensim)
*   Đã sử dụng Gensim để huấn luyện model **Word2Vec** mới trên tập dữ liệu nhỏ **`data/UD_English-EWT/en_ewt-ud-train.txt`**, theo kịch bản được mô tả trong `test/lab4_embedding_training_demo.py`.

#### 3. Advanced Task: Huấn luyện model trên tập dữ liệu lớn (Spark)
*   Đã cài đặt và cấu hình **PySpark** .
*   Sử dụng **Spark MLlib** để đọc, tiền xử lý (chuyển chữ thường, tokenization bằng `pyspark.ml.feature.Tokenizer`) và huấn luyện model Word2Vec trên dữ liệu lớn (ví dụ: `c4-train...json`), minh họa khả năng xử lý NLP ở quy mô lớn (scalability).

#### 4. Task 5: Trực quan hóa Embedding
*   Đã sử dụng kỹ thuật **Giảm chiều** (PCA hoặc t-SNE) để giảm các word vector (ví dụ: từ 50D xuống 2D).
*   Đã tạo biểu đồ scatter plot để trực quan hóa và quan sát mối quan hệ ngữ nghĩa giữa các cụm từ.

### C. Kết quả 

#### 1. So sánh Model Pre-trained (GloVe) và Model Tự huấn luyện (EWT)

Model GloVe pre-trained cho kết quả độ tương đồng và từ đồng nghĩa chính xác (ví dụ: $\text{king} - \text{man} + \text{woman} \approx \text{queen}$) do được huấn luyện trên dữ liệu khổng lồ (Wikipedia). Ngược lại, mô hình tự huấn luyện trên EWT nhỏ cho thấy kết quả kém:

*   **Các từ gần 'king' (EWT):** `[('assh', 0.7484), ('shedding', 0.7245), ('snakes', 0.7165), ('gyanendra', 0.7123), ('nepalese', 0.7073)]`. Kết quả này không phản ánh mối quan hệ ngữ nghĩa phổ quát, mà chỉ là **mối liên kết ngữ cảnh cục bộ** trong tập EWT.
*   **Phép suy luận ngữ nghĩa (EWT):** Phép toán $\text{king} - \text{man} + \text{woman}$ cho ra kết quả không liên quan (`['shedding', 'meat', 'neat']`). Điều này chứng tỏ tập dữ liệu nhỏ không đủ để mô hình Word2Vec học được các **mối quan hệ tuyến tính** trong không gian vector.

#### 2. Phân tích Biểu đồ trực quan hóa 

*   **Nhận xét:** Biểu đồ scatter plot 2D (sau khi giảm chiều) cho thấy các từ ngữ nghĩa tương đồng (ví dụ: từ chỉ nghề nghiệp, địa điểm) có xu hướng **tạo thành các cụm (cluster) riêng biệt**.
*   **Giải thích:** Kết quả này xác nhận nguyên lý Word Embedding: các từ chia sẻ ngữ cảnh sẽ nằm gần nhau trong không gian vector. Các kỹ thuật giảm chiều đã thành công trong việc bảo toàn mối quan hệ tương đồng ngữ nghĩa khi chiếu dữ liệu.

### D. Khó khăn

*   **Xử lý OOV:** Khó khăn khi tính document embedding (Task 3) nếu từ không có trong từ điển GloVe. **Giải pháp:** Triển khai logic kiểm tra sự tồn tại của từ. Các từ OOV bị bỏ qua khi tính trung bình, và nếu toàn bộ tài liệu là OOV, trả về vector không.
*   **Cấu hình Spark (Task 4):** Việc thiết lập PySpark yêu cầu cấu hình môi trường phức tạp. **Giải pháp:** Sử dụng Spark MLlib, tận dụng các hàm phân tán để xử lý dữ liệu lớn, đảm bảo khả năng mở rộng (scalability) .
### E. Tham khảo 
1. Gensim Documentation. 
  *Using Pretrained Word Embeddings and KeyedVectors.* 
  - Tutorial: [https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html) 
  - API Reference: [https://radimrehurek.com/gensim/models/keyedvectors.html](https://radimrehurek.com/gensim/models/keyedvectors.html)


2. PySpark MLlib Documentation. 
  *Word2Vec API for feature learning of word embeddings.* 
  - Link: [https://spark.apache.org/docs/latest/ml-features.html#word2vec](https://spark.apache.org/docs/latest/ml-features.html#word2vec)


3. scikit-learn Documentation. 
  *Dimensionality reduction techniques: PCA* 
  - PCA: [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) 