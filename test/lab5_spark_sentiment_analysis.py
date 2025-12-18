import sys
import os
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col

def run_spark_analysis():
    print("=== TASK 3: SPARK SENTIMENT ANALYSIS ===")
    
    # 1. Initialize Spark [cite: 288]
    spark = SparkSession.builder \
        .appName("SentimentAnalysis") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # 2. Load Data [cite: 291-297]
    data_path = "data/sentiments.csv"
    if not os.path.exists(data_path):
        print("Data file not found!")
        return

    df = spark.read.csv(data_path, header=True, inferSchema=True)
    # Convert sentiment -1/1 to 0/1 label (mapping: -1->0, 1->1)
    # Logic trong bài: (sentiment + 1) / 2 -> (-1+1)/2=0; (1+1)/2=1 [cite: 294]
    df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
    df = df.dropna(subset=["sentiment"])

    # Split data
    trainingData, testData = df.randomSplit([0.8, 0.2], seed=42)

    # 3. Build Pipeline [cite: 298-308]
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=1000)
    idf = IDF(inputCol="raw_features", outputCol="features")
    
    # 4. Model [cite: 312]
    lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")

    # Pipeline assembly [cite: 314]
    pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])

    # 5. Train & Evaluate [cite: 316-322]
    print("Training Spark Model...")
    model = pipeline.fit(trainingData)
    
    print("Predicting...")
    predictions = model.transform(testData)
    predictions.select("text", "label", "prediction").show(truncate=False)

    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"Spark Model Accuracy: {accuracy}")

    spark.stop()

if __name__ == "__main__":
    run_spark_analysis()