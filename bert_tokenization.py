# from transformers import BertTokenizerFast
# import torch
# from torch.utils.data import Dataset, DataLoader

# #initialize the BERT tokenizer
# tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# sentences = [
#     "We are looking for a Senior Data Scientist with experience in Python and SQL.",
#     "We are seeking a Full Stack Developer proficient in JavaScript and React.",
#     "Hiring a DevOps Engineer with expertise in AWS and Docker.",
#     "Looking for a Network Administrator familiar with Cisco systems and network security.",
#     "We need a Machine Learning Engineer experienced in TensorFlow and PyTorch.",
#     "Seeking a Cybersecurity Analyst with knowledge of penetration testing and firewall management.",
#     "Experience using analytical, marketing, and productivity tools including Oracle Business Intelligence, Salesforce or other CRM tools, Microsoft OneNote, and Microsoft SharePoint.",
#     "Possess strong data analytics (chart, dashboard etc) and coding skills with good understanding of IT Infrastructure concepts.",
#     "Familiar with cloud infrastructure, good understanding of different data storages and message queues for data streaming and pipelining.",
#     "Experience in machine learning frameworks (scikit-learn, Tensorflow, Pytorch), big data frameworks (Spark/Hadoop/Flink) and experience in resource management and task scheduling for large scale distributed systems."
# ]

# labels = [
#     ["O", "O", "O", "O", "O", "B-JOB_TITLE", "I-JOB_TITLE", "I-JOB_TITLE", "O", "O", "O", "B-SKILL", "O", "B-SKILL", "O"],
#     ["O", "O", "O", "O", "B-JOB_TITLE", "I-JOB_TITLE", "I-JOB_TITLE", "O", "O", "B-SKILL", "O", "B-SKILL", "O"],
#     ["O", "O", "O", "B-JOB_TITLE", "I-JOB_TITLE", "O", "O", "B-SKILL", "O", "B-SKILL", "O"],
#     ["O", "O", "O", "B-JOB_TITLE", "I-JOB_TITLE", "O", "O", "B-SKILL", "I-SKILL", "O", "B-SKILL", "I-SKILL", "O"],
#     ["O", "O", "O", "B-JOB_TITLE", "I-JOB_TITLE", "I-JOB_TITLE", "O", "O", "B-SKILL", "O", "B-SKILL", "O"],
#     ["O", "O", "B-JOB_TITLE", "I-JOB_TITLE", "O", "O", "O", "B-SKILL", "I-SKILL", "O", "B-SKILL", "I-SKILL", "O"],
#     ["O", "O", "B-SKILL", "O", "B-SKILL", "O", "O", "B-SKILL", "O", "O", "B-SKILL", "I-SKILL", "I-SKILL", "O", "B-SKILL", "O", "O", "B-SKILL", "O", "O", "B-SKILL", "I-SKILL", "O", "O", "B-SKILL", "I-SKILL", "O"],
#     ["O", "O", "B-SKILL", "I-SKILL", "O", "B-SKILL", "O", "B-SKILL", "O", "O", "O", "B-SKILL", "O", "O", "O", "O", "O", "B-TECH_TERM", "I-TECH_TERM", "O", "O"],
#     ["O", "O", "B-SKILL", "I-SKILL", "O", "O", "O", "O", "O", "B-SKILL", "I-SKILL", "O", "B-SKILL", "I-SKILL", "O", "B-SKILL", "I-SKILL", "O", "B-SKILL", "O"],
#     ["O", "O", "B-SKILL", "I-SKILL", "I-SKILL", "O", "B-SKILL", "O", "B-SKILL", "O", "B-SKILL", "O", "O", "B-SKILL", "I-SKILL", "I-SKILL", "O", "B-SKILL", "O", "B-SKILL", "O", "B-SKILL", "O", "O", "O", "O", "B-SKILL", "I-SKILL", "O", "B-SKILL", "I-SKILL", "O", "O", "O", "O", "O", "O"]
# ]

# #tokenize sentences and align labels
# def tokenize_and_align_labels(sentences, labels):
#     tokenized_inputs = tokenizer(sentences, truncation = True, padding = True, is_split_into_words = True, return_tensors = "pt",  return_offsets_mapping = True)
#     # Debug: Print tokenized inputs
#     print("Tokenized Inputs:", tokenized_inputs)
    
#     label_ids = []
    
#     # for i, label in enumerate(labels):
#     #     try: 
#     #         word_ids = tokenized_inputs.word_ids(batch_index = i)
#     #         # Debug: Print word_ids
#     #         print(f"Word IDs for sentence {i}:", word_ids)

#     #         aligned_labels = []
#     #         previous_word_id = None
#     #         for word_id in word_ids:
#     #             if word_id is None:
#     #                 aligned_labels.append(-100)
#     #             elif word_id != previous_word_id:
#     #                 aligned_labels.append(label[word_id])
#     #             else:
#     #                 aligned_labels.append(-100)
#     #             previous_word_id = word_id
#     #         label_ids.append(aligned_labels)
#     #     except IndexError as e:
#     #         print(f"IndexError for sentence {i}: {e}")
#     #         label_ids.append([-100] * len(tokenized_inputs.tokens(batch_index=i)))
    
#     # return tokenized_inputs, label_ids

#     for i, label in enumerate(labels):
#         word_ids = tokenized_inputs.word_ids(batch_index = i)
    
#         aligned_labels = [-100] * len(tokenized_inputs.input_ids[i])
        
#         previous_word_id = None
#         for j, word_id in enumerate(word_ids):
#             if word_id is not None:
#                 if word_id != previous_word_id:
#                     aligned_labels[j] = label[word_id]
#                 else:
#                     aligned_labels[j] = -100
#                 previous_word_id = word_id
        
#         label_ids.append(aligned_labels)
    
#     return tokenized_inputs, label_ids

# #tokenize and aling labels 
# tokenized_inputs, label_ids = tokenize_and_align_labels(sentences, labels)

# #convert label to tensors
# label_tensors = torch.tensor(label_ids)

# class NERDataset(Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item

#     def __len__(self):
#         return len(self.labels)

# # print("Tokenized Inputs:", tokenized_inputs)
# # print("Label IDs:", label_ids)

# #create dataset and dataLoader
# dataset = NERDataset(tokenized_inputs, label_tensors)
# loader = DataLoader(dataset, batch_size = 2, shuffle = True)

# #test
# batch = next(iter(loader))
# print(batch)