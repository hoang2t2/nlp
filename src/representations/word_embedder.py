import gensim.downloader as api
import numpy as np
from typing import List

from src.preprocessing.simple_tokenizer import SimpleTokenizer

class WordEmbedder:
    def __init__(self, model_name: str = 'glove-wiki-gigaword-50'):
        print(f"Loading model {model_name}...")
        # Task 1: Tải và sử dụng model có sẵn 
        self.model = api.load(model_name)
        self.tokenizer = SimpleTokenizer()
        print("Model loaded successfully.")

    def get_vector(self, word: str):
        
        if word in self.model:
            return self.model[word]
        else:
            # Xử lý OOV (Out-of-Vocabulary) bằng vector 0
            return np.zeros(self.model.vector_size)

    def get_similarity(self, word1: str, word2: str) -> float:
     
        if word1 in self.model and word2 in self.model:
            return self.model.similarity(word1, word2)
        return 0.0

    def get_most_similar(self, word: str, top_n: int = 10):

        if word in self.model:
            return self.model.most_similar(word, topn=top_n)
        return []

    def embed_document(self, document: str) -> np.ndarray:
        # Task 2: Nhúng văn bản (trung bình cộng vector)
        tokens = self.tokenizer.tokenize(document)
        vectors = []
        
        for token in tokens:
            if token in self.model:
                vectors.append(self.model[token])
        
        if not vectors:
            return np.zeros(self.model.vector_size)
        
        # Tính trung bình (element-wise mean)
        return np.mean(vectors, axis=0)