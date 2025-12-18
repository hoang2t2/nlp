import sys
import os
# Fix path để import được src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gensim.models import Word2Vec
from gensim.test.utils import datapath
from src.preprocessing.simple_tokenizer import SimpleTokenizer

def train_custom_model():
    print("=== TASK 3: TRAIN GENSIM MODEL ===")
    
    # 1. Đọc dữ liệu thô [cite: 98]
    data_path = "data/UD_English-EWT/en_ewt-ud-train.txt"
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Tiền xử lý (Tokenize)
    tokenizer = SimpleTokenizer()
    # Tách thành list các câu, mỗi câu là list các token
    sentences = [tokenizer.tokenize(line) for line in text.split('\n') if line.strip()]

    # 3. Huấn luyện model [cite: 13, 99]
    print("Training Word2Vec model...")
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)

    # 4. Lưu model [cite: 13, 100]
    os.makedirs("results", exist_ok=True)
    model.save("results/word2vec_ewt.model")
    print("Model saved to results/word2vec_ewt.model")

    # 5. Test thử
    word = "the"
    if word in model.wv:
        print(f"Vector for '{word}': {model.wv[word][:5]}...")
        print(f"Most similar to '{word}': {model.wv.most_similar(word, topn=3)}")

if __name__ == "__main__":
    train_custom_model()