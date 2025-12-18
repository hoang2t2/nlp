# Báo cáo Lab 7: Phân tích Cú pháp Phụ thuộc với spaCy

Báo cáo này trình bày các bước thực hiện và kết quả của bài thực hành chi tiết về **Dependency Parsing (Phân tích cú pháp phụ thuộc) với spaCy** .

## I. Giới thiệu và Cài đặt

Bài lab này tập trung vào kỹ thuật **Phân tích cú pháp phụ thuộc (Dependency Parsing)**, nhằm xác định cấu trúc ngữ pháp của câu bằng cách mô hình hóa quan hệ giữa các từ theo cặp **head – dependent**.

*   **Môi trường:** Đã sử dụng thư viện **spaCy** .
*   **Mô hình:** Đã tải và khởi tạo mô hình tiếng Anh kích thước trung bình: **`en_core_web_md`**. Mô hình này chứa đầy đủ thông tin cần thiết cho dependency parsing và word vectors .

## II. Kết quả Phân tích và Trích xuất Cấu trúc

### 1. Phân tích Cây Phụ thuộc (Dependency Tree)

*   **Câu ví dụ:** "The quick brown fox jumps over the lazy dog." .
*   **Từ gốc (ROOT):** Động từ chính của câu là **`jumps`** .
*   **Quan hệ phụ thuộc chính:**
    *   Token **`fox`**: Có quan hệ Chủ ngữ (`nsubj`) với `jumps` .
    *   Token **`over`**: Có quan hệ Giới từ bổ nghĩa (`prep`) với `jumps` .

### 2. Trích xuất Dependency của từng Token

Phân tích câu: "Apple is looking at buying U.K. startup for $1 billion" .

| Token | Dependency (`dep_`) | Head | Head POS |
| :---: | :---: | :---: | :---: |
| **looking** | ROOT | looking | VERB |
| **buying** | pcomp | at | ADP |
| **startup** | dobj | buying | VERB |
| **billions** | pobj | for | ADP |

Kết quả cho thấy **ROOT** của câu là **`looking`** .

### 3. Trích xuất Quan hệ Chủ ngữ – Động từ – Tân ngữ (S-V-O)

Đã triển khai logic để tìm các token là `nsubj` (Chủ ngữ) và `dobj` (Tân ngữ trực tiếp) của một động từ (`VERB`) [4].

*   **Câu test:** "The cat chased the mouse and the dog watched them." .
*   **Kết quả:**
    *   **cat chased mouse** .
    *   **dog watched them** .

### 4. Trích xuất Tính từ Bổ nghĩa Danh từ

Đã triển khai logic tìm các token con có quan hệ **`amod`** (adjectival modifier) với token cha là danh từ (`NOUN`) .

*   **Câu test:** "The big, fluffy white cat is sleeping on the warm mat." .
*   **Kết quả:**
    *   Danh từ: **`cat`** được bổ nghĩa bởi: **`['big', 'white']`**  hoặc **`['big', 'fluffy', 'white']`** .
    *   Danh từ: **`mat`** được bổ nghĩa bởi: **`['warm']`** .

### 5. Tìm Đường đi từ Token đến ROOT

Đã triển khai hàm `get_path_to_root(token)` để tìm đường đi từ một token bất kỳ đến token gốc của câu .

*   **Câu test 1:** "Apple is looking at buying U.K. startup for $1 billion".
    *   **Start token:** `startup` (index 6).
    *   **Đường đi tìm được:** **`['startup', 'buying', 'at', 'looking']`**.
*   **Câu test 2:** "The big, fluffy white cat is sleeping on the warm mat.".
    *   **Start token:** `white` (index 3).
    *   **Đường đi tìm được:** **`['white', 'cat', 'sleeping']`**.

## III. Kết luận

Qua buổi thực hành, các sinh viên đã đạt được các mục tiêu sau:

*   Hiểu về cây phụ thuộc (`dependency tree`) và ý nghĩa quan hệ `head–dependent`.
*   Xác định được **ROOT** (Động từ chính) của câu .
*   Thực hiện thành công việc trích xuất quan hệ S-V-O và tìm bổ nghĩa của danh từ thông qua amod .
*   Thực hiện việc truy vết (lần theo quan hệ) đến ROOT để phân tích cấu trúc câu.

Các ứng dụng thực tế của kỹ thuật này bao gồm: Trích xuất quan hệ (Relation Extraction), hệ thống hỏi đáp, phân tích ngữ nghĩa, và khai thác dữ liệu pháp luật.