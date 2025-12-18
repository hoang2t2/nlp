# Báo cáo Lab 7: Phân tích Cú pháp Phụ thuộc (Dependency Parsing) với spaCy

Lab này tập trung vào kỹ thuật Dependency Parsing, sử dụng thư viện `spaCy` để xác định cấu trúc ngữ pháp của câu thông qua quan hệ head – dependent.

## I. Giới thiệu và Cài đặt

### A. Cài đặt và Tải mô hình

```python
# 1. Cài đặt thư viện
# pip install -U spacy
# spacy download en_core_web_md

# 2. Tải mô hình và khởi tạo
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_md")
II. Phân tích và Trực quan hóa Cây Phụ thuộc
A. Phân tích Cấu trúc Token
Câu ví dụ: "The quick brown fox jumps over the lazy dog."
text = "The quick brown fox jumps over the lazy dog."
doc = nlp(text)

# Phân tích token: dep_, head, children 
print(f"Token: {doc[2].text}, DEP: {doc[2].dep_}, HEAD: {doc[2].head.text}, HEAD POS: {doc[2].head.pos_}")
# Kết quả: Token: jumps, DEP: ROOT, HEAD: jumps, HEAD POS: VERB
Kết quả Phân tích:
• Từ gốc (ROOT): jumps (động từ chính của câu).
• Quan hệ phụ thuộc chính:
    ◦ fox: Chủ ngữ (nsubj) của jumps.
    ◦ over: Giới từ bổ nghĩa (prep) cho jumps.
B. Phân tích Chi tiết Từng Token
Câu ví dụ: "Apple is looking at buying U.K. startup for $1 billions".
| Token | DEP | HEAD | HEAD POS | CHILDREN |
| :---: | :---: | :---: | :---: | :---: |
| Apple | nsubj | looking | VERB | [ ] |
| is | aux | looking | VERB | [ ] |
| looking | ROOT | looking | VERB | [Apple, is, at] |
| at | prep | looking | VERB | [buying] |
| buying | pcomp | at | ADP | [startup] |
| U.K. | compound | startup | NOUN | [ ] |
| startup | dobj | buying | VERB | [U.K., for] |
| for | prep | startup | NOUN | [billions] |
| $ | quantmod | billions | NOUN | [ ] |
| 1 | compound | billions | NOUN | [ ] |
| billions | pobj | for | ADP | [$, 1] |
III. Duyệt Cây Phụ thuộc để Trích xuất Thông tin
1. Tìm Chủ ngữ – Động từ – Tân ngữ (S-V-O)
Hàm này tìm các token là nsubj (chủ ngữ) và dobj (tân ngữ trực tiếp) của một động từ (VERB).
def extract_subject_verb_object(doc):
    for token in doc:
        # 1. Tìm động từ
        if token.pos_ == 'VERB':
            verb = token.text
            subject = ''
            obj = ''

            # 2. Duyệt các token con (children)
            for child in token.children:
                # Tìm Chủ ngữ (nsubj)
                if child.dep_ == 'nsubj':
                    subject = child.text
                # Tìm Tân ngữ trực tiếp (dobj)
                if child.dep_ == 'dobj':
                    obj = child.text
            
            # 3. In kết quả nếu tìm thấy cả hai
            if subject and obj:
                print(f'Found Triplet: ({subject}, {verb}, {obj})')

# Kết quả :
# Câu test: "The cat chased the mouse and the dog watched them."
# Found Triplet: (cat, chased, mouse)
# Found Triplet: (dog, watched, them)
2. Tìm Tính từ Bổ nghĩa cho Danh từ
Hàm này tìm các token con có quan hệ amod (adjectival modifier) với token cha là danh từ (NOUN).
def find_adjective_modifiers(doc):
    for token in doc:
        # 1. Tìm token là Danh từ (NOUN)
        if token.pos_ == 'NOUN':
            adjectives = []
            
            # 2. Duyệt các token con
            for child in token.children:
                # Tìm quan hệ bổ nghĩa tính từ
                if child.dep_ == 'amod': # adjectival modifier
                    adjectives.append(child.text)
            
            # 3. In kết quả
            if adjectives:
                # Kết quả có thể là danh sách [big, fluffy, white] 
                print(f"Danh từ: '{token.text}' được bổ nghĩa bởi: {adjectives}")

# Kết quả :
# Câu test: "The big, fluffy white cat is sleeping on the warm mat."
# Danh từ: 'cat' được bổ nghĩa bởi: ['big', 'fluffy', 'white']
# Danh từ: 'mat' được bổ nghĩa bởi: ['warm']
3. Tìm Đường đi từ Token đến ROOT 
Hàm này tìm đường đi từ một token cụ thể đến token gốc của câu (dep_ == 'ROOT').
• Hướng tiếp cận: Lần theo quan hệ head (parent) cho đến khi gặp token có dep_ == 'ROOT'.
def get_path_to_root(token):
    path = [token.text]
    current = token
    # Lặp cho đến khi gặp token gốc
    while current.dep_ != 'ROOT':
        current = current.head
        path.append(current.text)
        # Ngăn ngừa loop vô hạn trong trường hợp lỗi parsing
        if len(path) > len(token.doc):
            return "Path not found (potential cycle)"
    return path

# Kết quả thực nghiệm:
# Câu test: "The big, fluffy white cat is sleeping on the warm mat."
# Start token: white
# Đường đi tìm được: ['white', 'cat', 'sleeping']
# Giải thích: white (dependent) → cat (head) → sleeping (ROOT)
