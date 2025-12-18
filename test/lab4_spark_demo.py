import sys
import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, Word2Vec
from pyspark.sql.functions import col, lower, regexp_replace

# [cite: 118-122]

def run_spark_training():
    print("=== TASK 4: TRAIN SPARK MODEL ===")
    
    # 1. Setup Spark [cite: 15, 124]
    spark = SparkSession.builder \
        .appName("Word2VecExample") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # 2. Load data [cite: 16, 126]
    # Sử dụng file dummy json vừa tạo
    data_path = "data/dummy_data.json" 
    if not os.path.exists(data_path):
        print(f"Please create {data_path} first.")
        return

    df = spark.read.json(data_path)

    # 3. Preprocessing [cite: 131-134]
    # Lowercase & Remove punctuation
    df_clean = df.select(lower(col("text")).alias("text"))
    df_clean = df_clean.select(regexp_replace(col("text"), "[^a-zA-Z\\s]", "").alias("text"))
    
    # Tokenize
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    docData = tokenizer.transform(df_clean)

    # 4. Train Word2Vec [cite: 17, 136]
    word2Vec = Word2Vec(vectorSize=50, minCount=0, inputCol="words", outputCol="result")
    model = word2Vec.fit(docData)

    # 5. Show results
    print("Finding synonyms for 'learning':")
    try:
        synonyms = model.findSynonyms("learning", 5)
        synonyms.show()
    except Exception as e:
        print("Word 'learning' not found (dataset too small).")

    spark.stop()

if __name__ == "__main__":
    run_spark_training()