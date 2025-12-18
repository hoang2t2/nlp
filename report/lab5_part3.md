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
