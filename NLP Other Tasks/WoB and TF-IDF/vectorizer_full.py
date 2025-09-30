# vectorizer_full.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

# Create output folder
os.makedirs("output/bow_sentences", exist_ok=True)
os.makedirs("output/tfidf_sentences", exist_ok=True)

# 1. Corpus
corpus = [
    "Sam eats pizza after football.",
    "Pizza and burgers are delicious.",
    "Devi plays football on Sunday.",
    "Burgers and pizza after game.",
    "She loves pizza and tennis."
]

# 2. ---------------------- Bag of Words ----------------------
print("\n====== Bag of Words ======\n")

bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(corpus).toarray()

# 2.1 Vocabulary
print("Vocabulary:\n", bow_vectorizer.vocabulary_)

# 2.2 Matrix
print("\nBoW Matrix:")
for row in bow_matrix:
    formatted = ["{0}".format(val) for val in row]
    print("[{}]".format(" ".join(formatted)))

# 2.3 Sentence-wise WordClouds (BoW)
for i, sent in enumerate(corpus):
    vec = CountVectorizer()
    vec.fit(corpus)
    X = vec.transform([sent]).toarray()[0]
    word_freq = dict(zip(vec.get_feature_names_out(), X))
    wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
    wc.to_file(f"output/bow_sentences/bow_wordcloud_s{i+1}.png")

# 2.4 Cosine Similarity (BoW)
cos_bow = cosine_similarity(bow_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(cos_bow, annot=True, cmap="Blues", xticklabels=[f"S{i+1}" for i in range(5)],
            yticklabels=[f"S{i+1}" for i in range(5)])
plt.title("BoW Cosine Similarity")
plt.savefig("output/bow_similarity.png")
plt.close()

# 3. ---------------------- TF-IDF ----------------------
print("\n====== TF-IDF ======\n")

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus).toarray()

# 3.1 Vocabulary
print("Vocabulary:\n", tfidf_vectorizer.vocabulary_)

# 3.2 TF-IDF Matrix
print("\nTF-IDF Matrix:")
for row in tfidf_matrix:
    formatted = ["{0:.8f}".format(val) for val in row]
    print("[{}]".format(" ".join(formatted)))

# 3.3 Sentence-wise WordClouds (TF-IDF)
for i, sent in enumerate(corpus):
    vec = TfidfVectorizer()
    vec.fit(corpus)
    X = vec.transform([sent]).toarray()[0]
    word_freq = dict(zip(vec.get_feature_names_out(), X))
    wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
    wc.to_file(f"output/tfidf_sentences/tfidf_wordcloud_s{i+1}.png")

# 3.4 Cosine Similarity (TF-IDF)
cos_tfidf = cosine_similarity(tfidf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(cos_tfidf, annot=True, cmap="Oranges", xticklabels=[f"S{i+1}" for i in range(5)],
            yticklabels=[f"S{i+1}" for i in range(5)])
plt.title("TF-IDF Cosine Similarity")
plt.savefig("output/tfidf_similarity.png")
plt.close()
