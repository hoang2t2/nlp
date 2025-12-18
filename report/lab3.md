```markdown
# Báo Cáo Lab 3: Trực Quan Hóa Word Embedding

## I. Mục tiêu
Mục tiêu của bài lab này là nghiên cứu và thực hành kỹ thuật biểu diễn ngữ nghĩa của từ dưới dạng vector dày đặc (Embedding). Cụ thể, bài lab tập trung vào:

1.  **Giảm chiều:** Sử dụng các kỹ thuật **PCA** (Principal Component Analysis) hoặc **t-SNE** (t-distributed Stochastic Neighbor Embedding) để giảm chiều các word vector (đang ở 50 chiều) xuống còn 2 chiều.
2.  **Trực quan hóa:** Vẽ biểu đồ **scatter plot** để hiển thị các từ trong không gian 2D, qua đó quan sát mối quan hệ ngữ nghĩa giữa các từ và đánh giá chất lượng của mô hình embedding.

## II. Cài đặt và Chuẩn bị Dữ liệu

### A. Cài đặt Thư viện

Các thư viện cần thiết cho việc xử lý vector, giảm chiều và trực quan hóa đã được cài đặt, bao gồm `gensim` (cho mô hình embedding), `matplotlib` (cho vẽ biểu đồ), `scikit-learn` (cho PCA và t-SNE), và `numpy`.

### B. Tải Mô hình Pre-trained

Mô hình Word Embedding được sử dụng là **GloVe** (Global Vectors for Word Representation) đã được huấn luyện trước (`glove-wiki-gigaword-50`). Mô hình này cung cấp các vector 50 chiều cho mỗi từ.

### C. Chuẩn bị Tập Dữ liệu Trực quan hóa

Đã chọn 4 nhóm từ khác nhau để kiểm tra khả năng phân cụm ngữ nghĩa của mô hình GloVe:

| Nhóm | Các Từ |
| :---: | :---: |
| Family | father, mother, brother, sister, grandfather, grandmother, son, daughter |
| Colors | red, blue, green, yellow, orange, purple, white, black |
| Animals | cat, dog, lion, tiger, monkey, bird, fish, elephant |
| Technology | computer, software, hardware, internet, wifi, keyboard, screen, mouse |

Tổng cộng, **32 từ** đã được trích xuất vector . Kích thước của ma trận vector đầu vào là **(32, 50)**, nghĩa là 32 từ với mỗi từ là vector 50 chiều .

## III. Phương pháp Triển khai (Giảm chiều)

Các vector 50 chiều đã được giảm xuống 2 chiều bằng hai kỹ thuật chính: PCA và t-SNE.

### A. Giảm chiều với PCA

**PCA** (Principal Component Analysis) là một kỹ thuật tuyến tính tập trung vào việc tìm các chiều (trục) chứa nhiều phương sai nhất để giữ lại thông tin tổng thể của dữ liệu .

### B. Giảm chiều với t-SNE

**t-SNE** (t-distributed Stochastic Neighbor Embedding) là một kỹ thuật phi tuyến tính, được ưu tiên sử dụng để **trực quan hóa** vì nó nổi bật trong việc bảo toàn cấu trúc cục bộ (local structure), giúp các điểm gần nhau trong không gian gốc vẫn gần nhau trong không gian 2D . Do tập dữ liệu demo nhỏ (32 từ), `perplexity` đã được thiết lập thấp (5) .

## IV. Kết quả và Phân tích

### A. Trực quan hóa bằng PCA

Biểu đồ trực quan hóa bằng PCA cho thấy sự phân cụm ở mức độ tổng thể .

*   **Phân cụm thành công:** Các từ thuộc nhóm **'Family'** tập trung rõ ràng ở phía bên phải, và nhóm **'Technology'** tập trung ở phía bên trái của biểu đồ .
*   **Hạn chế:** Các nhóm **'Colors'** và **'Animals'** bị trộn lẫn và phân bố rải rác ở phần trên của biểu đồ, cho thấy PCA gặp khó khăn trong việc tách biệt các cụm có cấu trúc ngữ nghĩa phức tạp hoặc tinh tế hơn .

### B. Trực quan hóa bằng t-SNE

Biểu đồ t-SNE cho thấy hiệu quả trực quan hóa vượt trội .

*   **Phân cụm thành công:** Cả bốn nhóm từ đều được phân tách thành các cụm riêng biệt, rõ ràng và cô đọng .
    *   Nhóm **'Family'** tạo thành một cụm chặt chẽ ở phía trên cùng .
    *   Nhóm **'Colors'** tạo thành một cụm rõ rệt ở phía dưới bên trái .
    *   Nhóm **'Animals'** và **'Technology'** cũng được tách biệt thành hai cụm khác nhau .

### C. Nhận xét 

T-SNE chứng minh khả năng bảo toàn mối quan hệ ngữ nghĩa cục bộ của vector GloVe tốt hơn PCA . Trong không gian t-SNE, các từ có ngữ nghĩa tương đồng (như các thành viên trong gia đình, các màu sắc) được đặt rất gần nhau, làm nổi bật tính chất của Word Embedding là biểu diễn ngữ nghĩa của từ trong không gian vector: **các từ liên quan nằm gần nhau** . Bài lab đã hoàn thành mục tiêu chứng minh rằng Word Embedding có khả năng nắm bắt quan hệ ngữ nghĩa, và t-SNE là công cụ hiệu quả để trực quan hóa cấu trúc ngữ nghĩa này.
```