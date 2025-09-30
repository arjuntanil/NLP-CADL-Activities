# bow_wordclouds.py

import os
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Create output folder
os.makedirs("output/bow", exist_ok=True)

# Corpus
corpus = [
    "Sam eats pizza after football.",
    "Pizza and burgers are delicious.",
    "Devi plays football on Sunday.",
    "Burgers and pizza after game.",
    "She loves pizza and tennis."
]

# Vectorize with Bag-of-Words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
features = vectorizer.get_feature_names_out()
bow_matrix = X.toarray()

# Generate WordCloud for each sentence
for idx, vector in enumerate(bow_matrix):
    word_freq = {word: count for word, count in zip(features, vector) if count > 0}
    wc = WordCloud(width=600, height=400, background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(6, 4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"BoW WordCloud - Sentence {idx+1}")
    plt.savefig(f"output/bow/bow_sentence_{idx+1}.png")
    plt.close()

print("Bag-of-Words sentence-level WordClouds saved to output/bow/")
