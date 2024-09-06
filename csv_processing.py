import csv
import re

#load CSV
with open('dataset.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    sentences = []
    entities = []
    labels = []

    #print headers
    headers = reader.fieldnames
    print("CSV Headers:", headers)

    for row in reader:
        sentences.append(row['Sentence'])
        entities.append(row['Entity'].split(','))
        labels.append(row['Label'].split(','))

print(sentences)
print(entities)
print(labels)

#convert data to BIO format
bio_formatted_data = []

for sentence, entity_list, label_list in zip(sentences, entities, labels):
    sentence_words = re.findall(r'\w+|[^\w\s]', sentence)
    bio_labels = ['O'] * len(sentence_words)
    
    for entity, label in zip(entity_list, label_list):    
        entity_words = entity.split()
        entity_len = len(entity_words)
    
        for i in range(len(sentence_words) - entity_len + 1):
            if sentence_words[i:i+entity_len] == entity_words:
                bio_labels[i] = f'B-{label}'
                for j in range(1, entity_len):
                    bio_labels[i + j] = f'I-{label}'
                break
    
    bio_formatted_data.append(list(zip(sentence_words, bio_labels)))

#print BIO formatted data
for item in bio_formatted_data:
    print(item)