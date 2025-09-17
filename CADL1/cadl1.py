# ================================
# NLP Preprocessing in Python
# Using both NLTK and spaCy
# ================================



# Import & download resources
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')   
nltk.download('stopwords')
nltk.download('wordnet')

import spacy

nlp = spacy.load("en_core_web_sm")

# ================================
# Sample Dataset
# ================================
corpus = [
    "The stock market crashed due to global uncertainty.",
    "Natural Language Processing is a key part of Artificial Intelligence.",
    "Google releases a new AI model to improve search results.",
    "The weather today is sunny and pleasant in New York.",
    "Sports events are being postponed because of heavy rains."
]

print("📌 Original Corpus:")
for i, doc in enumerate(corpus, 1):
    print(f"{i}. {doc}")

# ================================
# 🔹 NLTK Preprocessing
# ================================
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print("\n================ NLTK Preprocessing ================")

for i, doc in enumerate(corpus, 1):
    tokens = word_tokenize(doc.lower())  # Tokenization
    no_stop = [w for w in tokens if w.isalpha() and w not in stop_words]  # Stopword removal
    stemmed = [stemmer.stem(w) for w in no_stop]  # Stemming
    lemmatized = [lemmatizer.lemmatize(w) for w in no_stop]  # Lemmatization

    print(f"\nSentence {i}: {doc}")
    print(f"👉 Tokens: {tokens}")
    print(f"👉 After Stopword Removal: {no_stop}")
    print(f"👉 After Stemming: {stemmed}")
    print(f"👉 After Lemmatization: {lemmatized}")

# ================================
# 🔹 spaCy Preprocessing
# ================================
print("\n================ spaCy Preprocessing ================")

for i, doc in enumerate(corpus, 1):
    spacy_doc = nlp(doc.lower())

    tokens = [token.text for token in spacy_doc]  # Tokenization
    no_stop = [token.text for token in spacy_doc if not token.is_stop and token.is_alpha]  # Stopword removal
    lemmatized = [token.lemma_ for token in spacy_doc if not token.is_stop and token.is_alpha]  # Lemmatization

    print(f"\nSentence {i}: {doc}")
    print(f"👉 Tokens: {tokens}")
    print(f"👉 After Stopword Removal: {no_stop}")
    print(f"👉 After Lemmatization: {lemmatized}")