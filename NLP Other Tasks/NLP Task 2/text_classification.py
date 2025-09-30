import pandas as pd
import re
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_excel("Dataset.xlsx")

# Preview
print("📘 Dataset Preview:\n")
print(df.head())

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text

df['clean_review'] = df['review'].apply(clean_text)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_review'], df['sentiment'], test_size=0.2, random_state=42
)

# === Bag-of-Words ===
bow_vectorizer = CountVectorizer(stop_words='english')
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

bow_model = LogisticRegression(max_iter=1000)
bow_model.fit(X_train_bow, y_train)
y_pred_bow = bow_model.predict(X_test_bow)

bow_accuracy = accuracy_score(y_test, y_pred_bow)
bow_report = classification_report(y_test, y_pred_bow)

print("\n🔤 BoW Classification Report:\n")
print(f"Accuracy: {bow_accuracy:.2f}")
print(bow_report)

# === TF-IDF ===
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

tfidf_model = LogisticRegression(max_iter=1000)
tfidf_model.fit(X_train_tfidf, y_train)
y_pred_tfidf = tfidf_model.predict(X_test_tfidf)

tfidf_accuracy = accuracy_score(y_test, y_pred_tfidf)
tfidf_report = classification_report(y_test, y_pred_tfidf)

print("\n📘 TF-IDF Classification Report:\n")
print(f"Accuracy: {tfidf_accuracy:.2f}")
print(tfidf_report)

# === Accuracy Bar Graph ===
methods = ['Bag-of-Words', 'TF-IDF']
accuracies = [bow_accuracy, tfidf_accuracy]

plt.figure(figsize=(8, 5))
bars = plt.bar(methods, accuracies, color=['cornflowerblue', 'lightgreen'])
plt.ylim(0, 1)
plt.title("Accuracy Comparison: Bag-of-Words vs TF-IDF")
plt.ylabel("Accuracy")
plt.xlabel("Vectorization Method")

for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig("accuracy_comparison.png")
plt.show()
