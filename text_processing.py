from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
from pyspark.sql import Row
# Установите ресурсы NLTK
nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download("stopwords")
nltk.download("wordnet")
# Настройка Spark-сессии
conf = SparkConf().setAppName("Text Processing").setMaster("local[*]")
sc = SparkContext(conf=conf)
spark = SparkSession.builder.config(conf=conf).getOrCreate()

print("Spark Session создана")



# Путь к текстовым файлам
data_dir = "/workspaces/lab5/workspace/"
texts = []
for file_name in os.listdir(data_dir):
    if file_name.endswith(".txt"):
        with open(os.path.join(data_dir, file_name), "r", encoding="utf-8") as file:
            texts.append(file.read())
from pyspark.sql.functions import udf




lemmatizer = WordNetLemmatizer()

# Функции обработки текста
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()

def tokenize_text(text):
    russian_stopwords  = set(stopwords.words("russian"))
    tokens = word_tokenize(text, language="russian")
    tokens = [token for token in tokens if token.isalnum()]  # Удаляем пунктуацию
    tokens = [token for token in tokens if token.lower() not in russian_stopwords]  # Удаляем стоп-слова
    return tokens

def lemmatize_text(tokens):
    return " ".join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])

# Создание UDF
clean_text_udf = udf(clean_text, StringType())
tokenize_text_udf = udf(lambda text: " ".join(tokenize_text(text)), StringType())
lemmatize_text_udf = udf(lambda text: lemmatize_text(word_tokenize(text)), StringType())

# Преобразование текстов в DataFrame
rows = [Row(text=text) for text in texts]
df = spark.createDataFrame(rows)

# Добавление колонок с обработанным текстом
df = df.withColumn("cleaned_text", clean_text_udf(df["text"]))
df = df.withColumn("tokenized_text", tokenize_text_udf(df["cleaned_text"]))
df = df.withColumn("lemmatized_text", lemmatize_text_udf(df["cleaned_text"]))

df.show(truncate=False)
rdd = df.rdd
print(rdd.take(5))
def cluster_by_length(text):
    length = len(text.split())
    if length < 50:
        return "short"
    elif length < 150:
        return "medium"
    else:
        return "long"

cluster_udf = udf(cluster_by_length, StringType())
df = df.withColumn("cluster", cluster_udf(df["lemmatized_text"]))
df.show(truncate=False)
df.write.csv("output/text_clusters.csv", header=True)
