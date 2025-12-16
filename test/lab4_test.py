import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.representations.word_embedder import WordEmbedder

def main():
    # Setup
    embedder = WordEmbedder() # Mặc định load glove-wiki-gigaword-50

    # 1. Test get_vector [cite: 90]
    print("\n--- Vector for 'king' ---")
    vec_king = embedder.get_vector('king')
    print(f"Shape: {vec_king.shape}, First 5 val: {vec_king[:5]}")

    # 2. Test similarity [cite: 91]
    print("\n--- Similarity ---")
    sim_kq = embedder.get_similarity('king', 'queen')
    sim_km = embedder.get_similarity('king', 'man')
    print(f"Similarity King-Queen: {sim_kq}")
    print(f"Similarity King-Man: {sim_km}")

    # 3. Test most similar [cite: 91]
    print("\n--- Most similar to 'computer' ---")
    print(embedder.get_most_similar('computer', top_n=5))

    # 4. Test Embed Document [cite: 92]
    print("\n--- Document Embedding ---")
    doc = "The queen rules the country."
    doc_vec = embedder.embed_document(doc)
    print(f"Doc: '{doc}'")
    print(f"Vector shape: {doc_vec.shape}, First 5 val: {doc_vec[:5]}")

if __name__ == "__main__":
    main()