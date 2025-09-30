from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import numpy as np

# Text data and labels (1 = positive, 0 = negative)
texts = [
    'I love this movie',
    'This movie is terrible',
    'I really enjoyed this film',
    'This film is awful',
    'What a fantastic experience',
    'I hated this film',
    'This was a great movie',
    'The film was not good',
    'I am very happy with this movie',
    'I am disappointed with this film'
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Positive sentiment, 0 = Negative sentiment

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the SVM model
model = SVC(kernel='linear')
model.fit(X_train_tfidf, y_train)

# Predict on test data
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("✅ Accuracy:", accuracy)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\n📊 Confusion Matrix:\n", conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"])
print("\n📄 Classification Report:\n", class_report)
