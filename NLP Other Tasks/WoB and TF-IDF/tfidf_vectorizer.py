# tfidf_wordclouds.py

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Create output folder
os.makedirs("output/tfidf", exist_ok=True)

# Corpus
corpus = [
    "Sam eats pizza after football.",
    "Pizza and burgers are delicious.",
    "Devi plays football on Sunday.",
    "Burgers and pizza after game.",
    "She loves pizza and tennis."
]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
tfidf_matrix = X.toarray()
features = vectorizer.get_feature_names_out()

# Sentence-level WordClouds
for idx, vector in enumerate(tfidf_matrix):
    word_freq = {word: tfidf for word, tfidf in zip(features, vector) if tfidf > 0}
    wc = WordCloud(width=600, height=400, background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(6, 4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"TF-IDF WordCloud - Sentence {idx+1}")
    plt.savefig(f"output/tfidf/tfidf_sentence_{idx+1}.png")
    plt.close()

print("✅ TF-IDF sentence-level WordClouds saved to output/tfidf/")

# Cosine Similarity Matrix
cos_sim_matrix = cosine_similarity(tfidf_matrix)

# Print Matrix with Diagonal = 1
print("\nCosine Similarity Matrix (diagonal should be 1):")
for row in cos_sim_matrix:
    print(["{:.2f}".format(val) for val in row])

# Visual Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cos_sim_matrix, annot=True, cmap="YlGnBu", xticklabels=[f"S{i+1}" for i in range(len(corpus))],
            yticklabels=[f"S{i+1}" for i in range(len(corpus))])
plt.title("TF-IDF Cosine Similarity Matrix")
plt.savefig("output/tfidf/tfidf_cosine_similarity.png")
plt.show()
