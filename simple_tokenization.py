import nltk
import spacy
from nltk.tokenize import word_tokenize

text = "This is a job description for a Python Developer."
tokens = word_tokenize(text)
print(tokens)

#load SpaCy model
nlp = spacy.load("en_core_web_trf")

#extract keywords
doc = nlp(text)
keywords = [ent.text for ent in doc.ents]
print("Keywords:", keywords)

#form search query
search_query = " AND ".join(keywords)
print(f"Generated Search Query: {search_query}")