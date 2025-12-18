import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.representations.word_embedder import WordEmbedder

def visualize_embeddings():
    print("=== TASK 5: VISUALIZATION ===")
    
    # Load model
    embedder = WordEmbedder()
    model = embedder.model

    # Chọn một số từ để vẽ
    words = ['king', 'queen', 'man', 'woman', 'apple', 'orange', 'fruit', 'computer', 'keyboard']
    valid_words = [w for w in words if w in model]
    
    if not valid_words:
        return

    # Lấy vector tương ứng
    vectors = [model[w] for w in valid_words]

    # Giảm chiều dữ liệu xuống 2D dùng PCA [cite: 19]
    pca = PCA(n_components=2)
    result = pca.fit_transform(vectors)

    # Vẽ biểu đồ Scatter [cite: 20]
    plt.figure(figsize=(10, 6))
    plt.scatter(result[:, 0], result[:, 1])

    for i, word in enumerate(valid_words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))

    plt.title("Word Embedding Visualization (PCA)")
    plt.grid(True)
    
    # Lưu biểu đồ để đưa vào báo cáo
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/embedding_plot.png")
    print("Plot saved to results/embedding_plot.png")
    plt.show()

if __name__ == "__main__":
    visualize_embeddings()