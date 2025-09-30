import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("Dataset.csv")

# Features and labels
X = df['review']
y = df['sentiment']

# Bag of Words Vectorization
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_bow, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Classification Report
print("🔍 Random Forest with Bag of Words")
print(classification_report(y_test, y_pred))
