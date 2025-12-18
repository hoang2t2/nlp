# Báo cáo Lab 6: Giới thiệu về Transformers
Báo cáo này trình bày các mục tiêu, kiến thức lý thuyết cơ bản, và kết quả thực hành của Lab 6, tập trung vào việc làm quen với kiến trúc Transformer và sử dụng thư viện `transformers` của Hugging Face để thực hiện các tác vụ NLP cơ bản.

## I.

### A. Mục tiêu Lab 6

Mục tiêu của Lab 6 là ôn tập kiến trúc **Transformer**, sử dụng các mô hình tiền huấn luyện (pretrained models) để thực hiện các tác vụ NLP cơ bản, và làm quen với thư viện `transformers`.

### B. Kiến trúc Transformer và các Mô hình liên quan

Kiến trúc Transformer bao gồm hai phần chính: **Encoder** và **Decoder**. Cơ chế cốt lõi là **Self-Attention** (tự chú ý), cho phép mô hình cân nhắc tầm quan trọng của các từ khác nhau trong câu khi xử lý một từ cụ thể, giúp nắm bắt các mối quan hệ ngữ nghĩa phức tạp.

## II. Bài tập Thực hành và Kết quả

Các bài tập được thực hiện bằng cách sử dụng `pipeline` của thư viện `transformers`.

### Bài 1: Khôi phục Masked Token (Masked Language Modeling)

**Yêu cầu:** Sử dụng pipeline `fill-mask` (sử dụng mô hình thuộc họ BERT) để dự đoán từ bị thiếu trong câu: `Hanoi is the [MASK] of Vietnam.` .

1.  **Mô hình đã dự đoán đúng từ `capital` không?**
    *   **Có**, mô hình đã dự đoán đúng từ "capital" với độ tin cậy rất cao (99%) [5]. Điều này chứng tỏ mô hình BERT đã học được mối quan hệ ngữ nghĩa giữa "Hanoi" và "capital of Vietnam" rất tốt [5].

2.  **Tại sao các mô hình Encoder-only như BERT lại phù hợp cho tác vụ này?**
    *   BERT sử dụng cơ chế **Self-Attention hai chiều (bidirectional)**, cho phép mô hình xem xét cả các từ đứng trước ("Hanoi is the") và các từ đứng sau ("[MASK] of Vietnam") của token `[MASK]` .
    *   BERT được huấn luyện trước với tác vụ **Masked Language Modeling (MLM)**, khiến nó rất giỏi trong việc dự đoán từ bị che giấu dựa trên ngữ cảnh xung quanh .
    *   Các mô hình Decoder-only (như GPT) không phù hợp vì chúng chỉ nhìn một chiều (từ trái sang phải) nên không thể tận dụng thông tin từ các từ phía sau token `[MASK]` .

### Bài 2: Dự đoán từ tiếp theo (Next Token Prediction)

**Yêu cầu:** Sử dụng pipeline `text-generation` (sử dụng mô hình thuộc họ GPT) để sinh ra phần tiếp theo cho câu: `The best thing about learning NLP is`.

1.  **Kết quả sinh ra có hợp lý không?**
    *   Kết quả sinh ra **thường khá hợp lý** về mặt ngữ pháp và có tính mạch lạc [6]. Mô hình GPT-2 mặc định được huấn luyện trên dữ liệu tiếng Anh nên nó có khả năng sinh văn bản tiếng Anh tự nhiên và có ngữ nghĩa .

2.  **Tại sao các mô hình Decoder-only như GPT lại phù hợp cho tác vụ này?**
    *   GPT được thiết kế với cơ chế **Self-Attention một chiều (unidirectional/causal)**, chỉ xem xét các token đã xuất hiện trước đó, điều này hoàn toàn phù hợp với tác vụ sinh văn bản tuần tự (từ trái sang phải) .
    *   GPT được huấn luyện với mục tiêu **Next Token Prediction** (dự đoán token tiếp theo) [8].
    *   Kiến trúc Decoder-only cho phép mô hình sinh từng token một cách **tự hồi quy (autoregressive)**, phù hợp với việc tạo ra chuỗi văn bản dài và mạch lạc .

### Bài 3: Tính toán Vector biểu diễn của câu (Sentence Representation)

**Yêu cầu:** Tính toán vector biểu diễn cho câu `This is a sample sentence.` bằng phương pháp **Mean Pooling** (trung bình cộng vector của tất cả các token, bỏ qua token đệm) .

1.  **Kích thước (chiều) của vector biểu diễn là bao nhiêu? Con số này tương ứng với tham số nào của mô hình BERT?**
    *   Kích thước vector biểu diễn tương ứng với tham số **`hidden_size`** của mô hình BERT-base. Từ kết quả, mỗi token được biểu diễn bằng vector **768 chiều** .

2.  **Tại sao chúng ta cần sử dụng `attention_mask` khi thực hiện Mean Pooling?**
    *   Chúng ta cần `attention_mask` để **loại bỏ padding tokens khỏi phép tính trung bình** .
    *   Khi xử lý batch nhiều câu có độ dài khác nhau, các câu ngắn hơn sẽ được đệm (padding) . Padding tokens không mang ý nghĩa ngữ nghĩa, và nếu tính chúng vào trung bình cộng sẽ làm méo mó vector biểu diễn ngữ nghĩa cuối cùng của câu .
