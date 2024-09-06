import spacy
import sklearn
from spacy.training import Example
from sklearn.model_selection import train_test_split

#load trained model
nlp = spacy.load("output/spacy_ner_bert_model")

#BIO data
data = [
    ("We are looking for a Senior Data Scientist with experience in Python and SQL.", [(21, 42, "JOB_TITLE"), (62, 68, "SKILL"), (73, 76, "SKILL")]),
    ("We are seeking a Full Stack Developer proficient in JavaScript and React.", [(17, 37, "JOB_TITLE"), (52, 62, "SKILL"), (67, 72, "SKILL")]),
    ("Hiring a DevOps Engineer with expertise in AWS and Docker.", [(9, 24, "JOB_TITLE"), (43, 46, "SKILL"), (51, 57, "SKILL")]),
    ("Looking for a Network Administrator familiar with Cisco systems and network security.", [(14, 35, "JOB_TITLE"), (50, 63, "SKILL"), (68, 84, "SKILL")]),
    ("We need a Machine Learning Engineer experienced in TensorFlow and PyTorch.", [(10, 35, "JOB_TITLE"), (51, 61, "SKILL"), (66, 73, "SKILL")]),
    ("Seeking a Cybersecurity Analyst with knowledge of penetration testing and firewall management.", [(10, 31, "JOB_TITLE"), (50, 69, "SKILL"), (74, 93, "SKILL")]),
    ("Experience using analytical, marketing, and productivity tools including Oracle Business Intelligence, Salesforce or other CRM tools, Microsoft OneNote, and Microsoft SharePoint.", [(17, 27, "SKILL"), (29, 38, "SKILL"), (44, 62, "SKILL"), (73, 101, "SKILL"), (103, 113, "SKILL"), (123, 132, "SKILL"), (134, 151, "SKILL"), (157, 177, "SKILL")]),
    ("Possess strong data analytics (chart, dashboard etc) and coding skills with good understanding of IT Infrastructure concepts", [(15, 29, "SKILL"), (57, 63, "SKILL"), (98, 124, "SKILL")]),
    ("Familiar with cloud infrastructure, good understanding of different data storages and message queues for data streaming and pipelining", [(14, 34, "SKILL"), (68, 81, "SKILL"), (86, 100, "SKILL"), (105, 119, "SKILL"), (124, 134, "SKILL")]),
    ("Experience in machine learning frameworks (scikit-learn, Tensorflow, Pytorch), big data frameworks (Spark/Hadoop/Flink) and experience in resource management and task scheduling for large scale distributed systems.", [(14, 41, "SKILL"), (43, 55, "SKILL"), (57, 67, "SKILL"), (69, 76, "SKILL"), (79, 98, "SKILL"), (100, 105, "SKILL"), (106, 112, "SKILL"), (113, 118, "SKILL"), (138, 157, "SKILL"), (162, 177, "SKILL"), (182, 213, "SKILL")])
]

#convert bio data to spacy format
train_data = []
for sentence, entities in data:
    doc = nlp.make_doc(sentence)
    annotations = {"entities": entities}
    example = Example.from_dict(doc, annotations)
    train_data.append(example)

#split data into training and validation set
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

#evaluate model
def evaluate_model(nlp, examples):
    scorer = nlp.evaluate(examples)
    print(f"Precision: {scorer['ents_p']:.3f}")
    print(f"Recall: {scorer['ents_r']:.3f}")
    print(f"F1-Score: {scorer['ents_f']:.3f}")

#evaluate using validation data
evaluate_model(nlp, val_data)

#test model
test_sentences = [
    "We are hiring a Data Engineer with knowledge of Scala and Hadoop.",
    "Looking for a Frontend Developer skilled in HTML, CSS, and JavaScript."
]

for sentence in test_sentences:
    doc = nlp(sentence)
    for ent in doc.ents:
        print(f"{ent.text}: {ent.label_}")

#fine tuning
optimizer = nlp.resume_training()
optimizer.learn_rate = 0.001  

#training for more iterations
for iteration in range(10): 
    losses = {}
    for example in train_data:
        nlp.update([example], sgd=optimizer, losses=losses)
    print(f"Iteration {iteration}: Losses {losses}")

#reevaluation
evaluate_model(nlp, val_data)

#saving fine tuned model
output_dir = "fine_tuned_ner_model"
nlp.to_disk(output_dir)
print(f"Fine-tuned model saved to {output_dir}")
