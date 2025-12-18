# Báo cáo Lab 1 (Text Tokenization) & Lab 2 (Count Vectorization)

Báo cáo này mô tả việc triển khai các thành phần nền tảng trong Xử lý Ngôn ngữ Tự nhiên (NLP): Tách từ (Tokenization) và Biểu diễn vector đếm (Count Vectorization), tuân thủ cấu trúc lập trình hướng đối tượng (OOP).

## I. Các bước triển khai 

Chúng tôi đã triển khai Lab 1 (Text Tokenization) và Lab 2 (Count Vectorization) như sau:

### A. Triển khai Lab 1: Text Tokenization

1.  **Định nghĩa Interface (src/core/interfaces.py):**
    *   Đã định nghĩa lớp cơ sở trừu tượng `Tokenizer` với phương thức bắt buộc `tokenize(self, text: str) -> list[str]`.
2.  **Triển khai SimpleTokenizer (src/preprocessing/simple_tokenizer.py):**
    *   Lớp này kế thừa từ `Tokenizer`. Phương thức `tokenize` thực hiện:
        *   Chuyển văn bản sang chữ thường .
        *   Tách token dựa trên khoảng trắng.
        *   Xử lý dấu câu cơ bản (. , ? !) bằng cách tách chúng khỏi từ .
3.  **Triển khai RegexTokenizer (src/preprocessing/regex_tokenizer.py):**
    *   Lớp này kế thừa từ `Tokenizer` và sử dụng một biểu thức chính quy để trích xuất token, nhằm xử lý các trường hợp phức tạp hơn.

### B. Triển khai Lab 2: Count Vectorization

Mục tiêu là biểu diễn văn bản dưới dạng vector số học bằng mô hình Bag-of-Words.

1.  **Định nghĩa Interface Vectorizer (src/core/interfaces.py):**
    *   Đã định nghĩa lớp cơ sở trừu tượng `Vectorizer` với các phương thức: `fit(self, corpus: list[str])`, `transform(self, documents: list[str]) -> list[list[int]]`, và `fit_transform`.
2.  **Triển khai CountVectorizer (src/representations/count_vectorizer.py):**
    *   Lớp này kế thừa từ `Vectorizer` và chấp nhận một instance `Tokenizer` (từ Lab 1) trong constructor `__init__(self, tokenizer: Tokenizer)`.
    *   **Phương thức `fit`:** Sử dụng tokenizer để xử lý corpus, thu thập các token duy nhất vào một tập hợp, sau đó xây dựng thuộc tính `vocabulary_` (một dictionary `str -> int`) để ánh xạ token sang chỉ mục .
    *   **Phương thức `transform`:** Tạo một vector không (zero vector) có độ dài bằng kích thước `vocabulary_`. Tokenize tài liệu và đếm tần số của các token có trong từ điển để điền vào vector.

## II. Chạy code và ghi log kết quả 

### A. Kết quả Lab 1: Text Tokenization (Ví dụ cơ bản)

Các tokenizer được kiểm thử với các câu mẫu theo yêu cầu :

| Câu mẫu | SimpleTokenizer (Output) | RegexTokenizer (Output) |
| :--- | :--- | :--- |
| **Original: Hello, world! This is a test.** | `['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']` | `['Hello', ',', 'world', '!', 'This', 'is', 'a', 'test', '.']` |
| **Original: NLP is fascinating... isn't it?** | `['nlp', 'is', 'fascinating', '.', '.', '.', "isn't", 'it', '?']` | `['NLP', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']` |
| **Original: Let's see how it handles 123 numbers and punctuation!** | `['let\'s', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']` | `['Let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']` |

### B. Kết quả Lab 1: Tokenization trên Dataset UD\_English-EWT

Áp dụng tokenizer lên 500 ký tự đầu tiên (`sample_text`) của dataset UD\_English-EWT :

```log
=== TEST 3: DATASET UD_ENGLISH-EWT ===
Original Sample (first 100 chars): Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the
mosque in the town of ...

SimpleTokenizer Output (first 20 tokens): ['al-zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al-ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the', 'town', 'of', 'qaim', ',']
RegexTokenizer Output (first 20 tokens): ['al', '-', 'zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', '-', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the']
```

### C. Kết quả Lab 2: Count Vectorization (Ví dụ cơ bản)
Sử dụng corpus mẫu: ["I love NLP.", "I love programming.", "NLP is a subfield of AI."]
Vocabulary Learned (vocabulary_):
{
    'i': 0, 'love': 1, 'nlp': 2, '.': 3, 'programming': 4,
    'is': 5, 'a': 6, 'subfield': 7, 'of': 8, 'ai': 9
}

Document-Term Matrix (Ma trận tần số từ):
[
  [3], # "I love NLP."
  [3], # "I love programming."
  [3]  # "NLP is a subfield of AI."
]


## III. Kết quả thu được 
### A. So sánh SimpleTokenizer và RegexTokenizer
1. Chuyển đổi chữ thường: SimpleTokenizer chuyển tất cả token về chữ thường ('hello', 'nlp') theo yêu cầu. Ngược lại, RegexTokenizer trong triển khai này giữ nguyên chữ hoa ở đầu câu ('Hello', 'NLP'), cho thấy nó tập trung vào việc tách dựa trên mẫu regex mà không thực hiện tiền xử lý chữ thường.
2. Xử lý Từ ghép/Dấu gạch nối:
    ◦ Trên dataset UD_English-EWT, SimpleTokenizer giữ 'al-zaman' và 'al-ani' là các token duy nhất.
    ◦ RegexTokenizer tách các từ này thành ba token riêng biệt: ['al', '-', 'zaman'] và ['al', '-', 'ani']. Điều này là kết quả của việc biểu thức chính quy coi dấu gạch nối (-) là một ký tự không phải từ ([^\w\s]), cho thấy tính chi tiết cao hơn của RegexTokenizer trong việc phân tách các thành phần cấu tạo từ.
### B. Phân tích CountVectorizer (Mô hình Bag-of-Words)
1. Vocabulary: Phương thức fit đã thành công trong việc xây dựng từ điển (vocabulary_) bằng cách ánh xạ các token duy nhất (nhận được từ tokenizer) sang chỉ mục số nguyên (ví dụ: 'i' → 0, 'love' → 1).
2. Document-Term Matrix (Ma trận tần số từ):
    ◦ Ma trận thể hiện mô hình Bag-of-Words, nơi mỗi hàng là một tài liệu được biểu diễn dưới dạng vector đếm. Các giá trị trong vector là tần suất xuất hiện của các từ trong từ điển.
    ◦ Ví dụ, tài liệu thứ ba ("NLP is a subfield of AI.") có tần suất đếm là 1 cho các từ 'nlp' (index 2), 'is' (index 5), 'a' (index 6),... và 0 cho các từ không xuất hiện như 'love' (index 1) hoặc 'programming' (index 4). Thứ tự từ hoàn toàn bị bỏ qua, chỉ có sự xuất hiện của chúng là quan trọng.
## IV. Khó khăn gặp phải và cách giải quyết 
Khó khăn 1: Xử lý Dấu nháy đơn trong RegexTokenizer:
    • Vấn đề: Biểu thức chính quy cơ bản (\w+|[^\w\s]) có xu hướng tách các từ viết tắt có dấu nháy đơn (ví dụ: "isn't" và "Let's") thành ba token ('isn', "', 't'). Điều này đôi khi làm mất đi ngữ nghĩa của token gốc.
    • Giải quyết: Đã chấp nhận kết quả tách chi tiết này để tuân thủ tính nhất quán của regex đã chọn, nhưng lưu ý rằng cần sử dụng một biểu thức chính quy phức tạp hơn (như \w+[']\w+|\w+|[^\w\s]) để giữ nguyên các từ viết tắt, nếu mục tiêu là giữ các token ngữ nghĩa hơn.
Khó khăn 2: Đảm bảo tính liên kết OOP giữa các Lab:
    • Vấn đề: Đảm bảo CountVectorizer sử dụng Tokenizer thông qua dependency injection (truyền instance vào constructor) và gọi đúng phương thức tokenize().
    • Giải quyết: Áp dụng cấu trúc OOP chặt chẽ, nơi CountVectorizer chỉ tương tác với interface Tokenizer, giúp code dễ dàng thay đổi bộ tokenizer (ví dụ: chuyển từ Simple sang Regex) mà không cần sửa đổi logic vectorization.

