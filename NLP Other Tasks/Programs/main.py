import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy

# Download necessary NLTK data (do this only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize NLTK tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# -------------------------------
# Function: Preprocess using NLTK
# -------------------------------
def preprocess_nltk(text):
    print("\n--- Using NLTK ---")

    # Tokenization
    tokens = word_tokenize(text)
    print("\nTokenized Words:")
    print(tokens)

    # Stopword Removal
    filtered = [word for word in tokens if word.lower() not in stop_words]
    print("\nAfter Stop Word Removal:")
    print(filtered)

    # Stemming
    stemmed = [stemmer.stem(word) for word in filtered]
    print("\nAfter Stemming:")
    print(stemmed)

    # Lemmatization
    lemmatized = [lemmatizer.lemmatize(word) for word in filtered]
    print("\nAfter Lemmatization:")
    print(lemmatized)


# -------------------------------
# Function: Preprocess using spaCy
# -------------------------------
def preprocess_spacy(text):
    print("\n--- Using spaCy ---")
    doc = nlp(text)

    # Tokenization
    tokens = [token.text for token in doc]
    print("\nTokenized Words:")
    print(tokens)

    # Stopword Removal
    filtered = [token for token in doc if not token.is_stop]
    print("\nAfter Stop Word Removal:")
    print([token.text for token in filtered])

    # Simulated Stemming (spaCy doesn’t have real stemming; using lowercased root form)
    pseudo_stemmed = [token.lemma_.lower() for token in filtered]
    print("\nAfter Simulated Stemming (via Lemma Lowercase):")
    print(pseudo_stemmed)

    # Lemmatization
    lemmatized = [token.lemma_ for token in filtered]
    print("\nAfter Lemmatization:")
    print(lemmatized)


# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    text = input("Enter your text: ")

    preprocess_nltk(text)
    preprocess_spacy(text)
