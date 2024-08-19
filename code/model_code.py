import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from datasets import Dataset
import os

# 0. GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

# 1. Load Dataset
def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path, encoding='utf-8')
    test_data = pd.read_csv(test_path, encoding='utf-8')

    train_texts = train_data['text'].tolist()
    train_labels = train_data['target'].tolist()
    test_texts = test_data['text'].tolist()
    test_labels = test_data['target'].tolist()

    return train_texts, test_texts, train_labels, test_labels

# 2. Train BERT-based model
def tokenize_data(texts, tokenizer, max_length=512):
    if isinstance(texts, list):
        texts = [str(text) if text is not None else "" for text in texts]
    else:
        texts = str(texts) if texts is not None else ""

    return tokenizer(texts, padding='max_length', truncation=True, return_tensors="pt", max_length=max_length)

def train_bert(llm_name, train_texts, train_labels, test_texts, test_labels, output_dir='./model/bert_lense'):
    if llm_name is None:
        llm_name = 'roberta-base'
    print("BERTLense is fine-tuned on", llm_name)
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    model = AutoModelForSequenceClassification.from_pretrained(llm_name, num_labels=2).to(device)

    train_encodings = tokenize_data(train_texts, tokenizer)
    test_encodings = tokenize_data(test_texts, tokenizer)

    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': torch.tensor(train_labels)
    })

    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels': torch.tensor(test_labels)
    })

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        logging_dir='./bert_logs',
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer, model, tokenizer

# 3. Train GPT-based model
def train_gpt(llm_name, train_texts, test_texts, output_dir='./model/gpt_lense'):
    if llm_name is None:
        llm_name = 'gpt2'
    print("GPTLense is fine-tuned on", llm_name)
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(llm_name).to(device)

    train_encodings = tokenize_data(train_texts, tokenizer)
    test_encodings = tokenize_data(test_texts, tokenizer)

    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_encodings['input_ids']
    })

    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels': test_encodings['input_ids']
    })

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        logging_dir='./gpt_logs',
        load_best_model_at_end=True,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

    # Save model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer, model, tokenizer

# 4. Load Model and Tokenizer
def load_model_and_tokenizer(model_dir, model_class):
    model = model_class.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

# 5. Fake News Detection
def FakeLense(text, bert_model, bert_tokenizer, gpt_model, gpt_tokenizer, similarity_threshold=0.8):
    bert_inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    bert_outputs = bert_model(**bert_inputs, output_hidden_states=True) # return hidden_states
    bert_prediction = torch.argmax(bert_outputs.logits, dim=1).item()

    #gpt_inputs = gpt_tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True).to(device)
    #gpt_outputs = gpt_model.generate(gpt_inputs, max_length=100)
    gpt_inputs = gpt_tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True).to(device)
    gpt_outputs = gpt_model.generate(gpt_inputs, max_length=100, pad_token_id=gpt_tokenizer.eos_token_id)
    generated_text = gpt_tokenizer.decode(gpt_outputs[0], skip_special_tokens=True)

    generated_bert_inputs = bert_tokenizer(generated_text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    generated_bert_outputs = bert_model(**generated_bert_inputs, output_hidden_states=True)

    # Extracting embeddings for cosine similarity
    bert_embedding = bert_outputs.hidden_states[-1][:,0,:]  # [CLS] token embedding from the last hidden layer
    generated_bert_embedding = generated_bert_outputs.hidden_states[-1][:,0,:]

    similarity = torch.nn.functional.cosine_similarity(bert_embedding, generated_bert_embedding, dim=1).item()

    if bert_prediction == 1 or similarity < similarity_threshold:
        return "Fake News Detected."
    else:
        return "Real News Detected."
    
    
# 6. Training Phase
train_texts, test_texts, train_labels, test_labels = load_data("./data/train.csv", "./data/test.csv")
gpt_trainer, gpt_lense, gpt_tokenizer = train_gpt('EleutherAI/gpt-neo-125M', train_texts, test_texts)
bert_trainer, bert_lense, bert_tokenizer = train_bert('microsoft/deberta-base', train_texts, train_labels, test_texts, test_labels)


# 7. Detection Phase
bert_lense, bert_tokenizer = load_model_and_tokenizer('./model/bert_lense', AutoModelForSequenceClassification)
gpt_lense, gpt_tokenizer = load_model_and_tokenizer('./model/gpt_lense', AutoModelForCausalLM)

# Test cases
test_cases = [
    # Truth News
    "Global Leaders Gather for Climate Summit: World leaders have convened in Paris for the annual climate summit, where they are expected to negotiate new commitments to reduce greenhouse gas emissions and combat climate change. The summit comes as scientists warn of increasingly severe weather patterns linked to global warming.",
    "The U.S. unemployment rate dropped to 3.6% in July, the lowest in over a decade, as the job market continues to recover.",

    # Fake News
    "Breaking News: The moon is made of cheese, claims new scientific report.",
    "World Health Organization confirms that drinking bleach can cure COVID-19, urges people to start treatment immediately.",
    "Scientists discover a new continent hidden beneath the Pacific Ocean, larger than Australia.",
    "All global currencies will become obsolete next week due to the launch of a universal cryptocurrency backed by all major governments.",
    "The sun is expected to explode within the next 24 hours, according to a secret NASA report leaked by an anonymous source."
]

for i, text in enumerate(test_cases):
    result = FakeLense(text, bert_lense, bert_tokenizer, gpt_lense, gpt_tokenizer)
    print(f"News {i+1} : {result}\n")
