from fake_lense import * 

# Training Phase
# Load Dataset
train_texts, test_texts, train_labels, test_labels = load_data("./data/train.csv", "./data/test.csv")

# BERTLense : Train on BERT-based model
bert_trainer, bert_lense, bert_tokenizer = train_bert(None, train_texts, train_labels, test_texts, test_labels, 1)
# GPTLense : Train on GPT-based model 
gpt_trainer, gpt_lense, gpt_tokenizer = train_gpt(None, train_texts, test_texts, 1)

# If user need to continue training
#gpt_trainer, gpt_lense, gpt_tokenizer = train_gpt(None, train_texts, test_texts, 1, True)
#bert_trainer, bert_lense, bert_tokenizer = train_bert(None, train_texts, train_labels, test_texts, test_labels, 1, True)
