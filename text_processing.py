import nltk
nltk.download('punkt')
nltk.download('wordnet')
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Настройка Spark
conf = SparkConf().setAppName("TextProcessing").setMaster("local[*]")
sc = SparkContext(conf=conf)
spark = SparkSession.builder.appName("TextPreprocessing").getOrCreate()

print("Spark session created!")

# Функции для предобработки текста
def clean_text(text):
    text = re.sub(r"[^\w\s]", "", text)  # Убираем пунктуацию
    return text.lower()

def tokenize_text(text):
    return " ".join(word_tokenize(text))

lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    tokens = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(token) for token in tokens])

def classify_text(text):
    if "ужас" in text:
        return "horror"
    elif "море" in text:
        return "adventure"
    else:
        return "unknown"

# Создание UDF
clean_udf = udf(clean_text, StringType())
tokenize_udf = udf(tokenize_text, StringType())
lemmatize_udf = udf(lemmatize_text, StringType())
classify_udf = udf(classify_text, StringType())

# Загрузка данных
data = spark.read.text("11-16.txt")
df = data.withColumnRenamed("value", "original_text")

# Предобработка текста
df = df.withColumn("cleaned_text", clean_udf(df.original_text))
df = df.withColumn("tokenized_text", tokenize_udf(df.cleaned_text))
df = df.withColumn("lemmatized_text", lemmatize_udf(df.tokenized_text))

# Кластеризация
df = df.withColumn("genre", classify_udf(df.cleaned_text))

# Сохранение результатов
df.write.format("csv").save("processed_texts.csv")

# Преобразование DataFrame в RDD (если нужно)
rdd = df.rdd
print(rdd.take(5))

print("Processing complete! Results saved to 'processed_texts.csv'")
