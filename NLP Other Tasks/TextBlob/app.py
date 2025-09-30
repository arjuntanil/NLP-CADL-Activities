# app.py

from textblob import TextBlob

# Get user input
text = input("Enter a sentence: ")

# Create TextBlob object
blob = TextBlob(text)

# Sentiment Analysis
print("\n🔍 Sentiment Analysis")
print("Polarity:", blob.sentiment.polarity)
print("Subjectivity:", blob.sentiment.subjectivity)

# Spelling Correction
print("\n📝 Spelling Correction")
print("Corrected Text:", blob.correct())

# Noun Phrases
print("\n🔍 Noun Phrases")
print(blob.noun_phrases)

# Part-of-Speech Tags
print("\n🔤 Part-of-Speech Tags")
print(blob.tags)
