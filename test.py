import spacy

#load fine tuned model
nlp = spacy.load("output/fine_tuned_ner_model")

test_sentences = [
    "Looking for a Data Scientist with strong Python skills.",
    "Hiring a DevOps expert with AWS and Docker experience."
]

for sentence in test_sentences:
    doc = nlp(sentence)
    print(f"Sentence: {sentence}")
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")
    print()
