from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.core.dataset_loaders.load_raw_text_data import load_raw_text_data



simple_tokenizer = SimpleTokenizer()
regex_tokenizer = RegexTokenizer()

sentences = [
    "Hello, world! This is a test.",
    "NLP is fascinating... isn't it?",
    "Let's see how it handles 123 numbers and punctuation!"
]

print("--- SimpleTokenizer ---")
for s in sentences:
    print(simple_tokenizer.tokenize(s))

print("\n--- RegexTokenizer ---")
for s in sentences:
    print(regex_tokenizer.tokenize(s))



dataset_path = "F:\\nlp\\nlp\\data\\UD_English-EWT\\UD_English-EWT\\en_ewt-ud-train.txt"
raw_text = load_raw_text_data(dataset_path)
sample_text = raw_text[:500]


print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
print(f"Original Sample: {sample_text[:100]}...")
print("SimpleTokenizer Output:", simple_tokenizer.tokenize(sample_text)[:20])
print("RegexTokenizer Output:", regex_tokenizer.tokenize(sample_text)[:20])
