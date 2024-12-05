from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .appName("TextProcessing") \
    .getOrCreate()
sc = spark.sparkContext
df = spark.read.text("path_to_your_text_data.txt")
df.show()

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import nltk
nltk.download('punkt')

def tokenize(text):
    return nltk.word_tokenize(text)

tokenize_udf = udf(tokenize, StringType())

df = spark.read.csv("your_text_file.csv", header=True)
df.show()

df = df.withColumn("tokens", tokenize_udf(df["text"]))
df.show()

rdd = df.rdd
rdd.collect()

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
idf = IDF(inputCol="rawFeatures", outputCol="features")

kmeans = KMeans().setK(3).setSeed(1)


tokenizer_model = tokenizer.fit(df)
df = tokenizer_model.transform(df)
hashingTF_model = hashingTF.fit(df)
df = hashingTF_model.transform(df)
idf_model = idf.fit(df)
df = idf_model.transform(df)

kmeans_model = kmeans.fit(df)
clustered_df = kmeans_model.transform(df)
clustered_df.show()
