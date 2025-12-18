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
