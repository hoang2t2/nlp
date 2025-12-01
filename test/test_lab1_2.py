import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.core.dataset_loaders.load_raw_text_data import load_raw_text_data



def run_tests():
    
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()
    
    test_sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!"
    ]

    # --- Task 1 & 2: Basic Tests  ---
    print("=== TEST 1&2: BASIC EXAMPLES ===")
    for text in test_sentences:
        print(f"\nOriginal: {text}")
        print(f"Simple: {simple_tokenizer.tokenize(text)}")
        print(f"Regex : {regex_tokenizer.tokenize(text)}")

    # --- Task 3: UD_English-EWT Dataset  ---
    print("\n=== TEST 3: DATASET UD_ENGLISH-EWT ===")
    
    dataset_path = "C:/Users/Hoang/nlp/data/UD_English-EWT/UD_English-EWT/en_ewt-ud-train.txt"
    
    try:
        raw_text = load_raw_text_data(dataset_path)
        sample_text = raw_text[:500] # Lấy 500 ký tự đầu
        
        print(f"Original Sample (first 100 chars): {sample_text[:100]}...")
        
        simple_tokens = simple_tokenizer.tokenize(sample_text)
        print(f"\nSimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")
        
        regex_tokens = regex_tokenizer.tokenize(sample_text)
        print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Check 'dataset_path'")

if __name__ == "__main__":
    run_tests()