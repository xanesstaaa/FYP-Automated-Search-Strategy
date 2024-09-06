import spacy
import random
from spacy.training import Example
from spacy_transformers import Transformer
from spacy.util import minibatch, compounding

#load spacy model
nlp = spacy.blank("en")

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

#add transformer component
nlp.add_pipe("transformer", config = {
    "model": {
        "@architectures": "spacy-transformers.TransformerModel.v3",
        "name": "bert-base-uncased",
        "tokenizer_config": {"use_fast": True}
    }
}, last = True)

#add ner
nlp.add_pipe("ner", last=True)

#add labels
ner = nlp.get_pipe("ner")
labels = ["JOB_TITLE", "SKILL"]
for label in labels:
    ner.add_label(label)

#initialize pipeline
optimizer = nlp.begin_training()

#training loop
for i in range(10): 
    losses = {}
    #shuffle training data
    random.shuffle(train_data)
    batches = minibatch(train_data, size = compounding(4.0, 32.0, 1.001))
    for batch in batches:
        nlp.update(batch, drop = 0.5, losses=losses)
    print(f"Iteration {i}: Losses {losses}")

#save model
nlp.to_disk("output/spacy_ner_bert_model")

#test model
doc = nlp("We are looking for a Senior Data Scientist with experience in Python and SQL.")
for ent in doc.ents:
    print(ent.text, ent.label_)