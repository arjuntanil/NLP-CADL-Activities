import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from wordcloud import WordCloud

# Step 1: Create the dataset
corpus = [
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

# Step 2: Labels for sentiment (1 = Positive, 0 = Negative)
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

# Step 3: Create 'Output' folder if not exists
os.makedirs("Output", exist_ok=True)

# Step 4: TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus)

# Step 5: Train using Multinomial Naive Bayes
model_tfidf = MultinomialNB()
model_tfidf.fit(X_tfidf, labels)
pred_tfidf = model_tfidf.predict(X_tfidf)

# Step 6: Confusion Matrix
cm_tfidf = confusion_matrix(labels, pred_tfidf)
print("\n=== Confusion Matrix (TF-IDF) ===")
print(cm_tfidf)

# Step 7: Evaluation Metrics
accuracy = accuracy_score(labels, pred_tfidf)
precision = precision_score(labels, pred_tfidf)
recall = recall_score(labels, pred_tfidf)
f1 = f1_score(labels, pred_tfidf)

print(f"\n✅ Accuracy:  {accuracy * 100:.2f}%")
print(f"✅ Precision: {precision * 100:.2f}%")
print(f"✅ Recall:    {recall * 100:.2f}%")
print(f"✅ F1 Score:  {f1 * 100:.2f}%")

# Step 8: Full Classification Report
print("\n=== Classification Report ===")
print(classification_report(labels, pred_tfidf, target_names=["Negative", "Positive"]))

# Step 9: Save confusion matrix as image
disp_tfidf = ConfusionMatrixDisplay(confusion_matrix=cm_tfidf, display_labels=["Negative", "Positive"])
fig, ax = plt.subplots(figsize=(6, 4))
disp_tfidf.plot(ax=ax, cmap="Blues", values_format='d')
plt.title("Confusion Matrix - TF-IDF")
plt.savefig("Output/confusion_matrix_tfidf.png")
plt.close()

# Step 10: Word Cloud for each sentence
for idx, sentence in enumerate(corpus):
    wc = WordCloud(width=600, height=300, background_color='white').generate(sentence)
    wc.to_file(f"Output/wordcloud_sentence_{idx+1}.png")

# Step 11: Combined Word Cloud
all_text = ' '.join(corpus)
combined_wc = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(combined_wc, interpolation='bilinear')
plt.axis("off")
plt.title("Combined Word Cloud")
plt.savefig("Output/wordcloud_combined.png")
plt.close()

# Step 12: Final Output Summary
print("\n Outputs saved in 'Output' folder:")
print("→ Confusion Matrix: Output/confusion_matrix_tfidf.png")
print("→ Combined Word Cloud: Output/wordcloud_combined.png")
for i in range(1, len(corpus)+1):
    print(f"→ Word Cloud (Sentence {i}): Output/wordcloud_sentence_{i}.png")
