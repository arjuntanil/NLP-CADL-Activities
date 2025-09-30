import spacy

# Load spaCy transformer-based English pipeline (uses BERT internally)
nlp = spacy.load("en_core_web_trf")

# Input text
text = "Barack Obama visited India in 2010."

# Process the text
doc = nlp(text)

# Print the named entities
print("Named Entities, their labels, and positions:")
for ent in doc.ents:
    print(f"{ent.text} → {ent.label_} (start: {ent.start_char}, end: {ent.end_char})")
