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

def tokenize_data(texts, tokenizer, max_length=512):
    if isinstance(texts, list):
        texts = [str(text) if text is not None else "" for text in texts]
    else:
        texts = str(texts) if texts is not None else ""

    return tokenizer(texts, padding='max_length', truncation=True, return_tensors="pt", max_length=max_length)

# 2. Load Model and Tokenizer
def load_model_and_tokenizer(model_dir, model_class):
    model = model_class.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

# 3. Train BERT-based model
def train_bert(llm_name, train_texts, train_labels, test_texts, test_labels, epochs, fine_tune=False, output_dir='./model/bert_lense'):
    if fine_tune and os.path.exists(output_dir):
        model, tokenizer = load_model_and_tokenizer(output_dir, AutoModelForSequenceClassification)
        print("BERTLense is fine-tuned on BERTLense again")
    else:
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
        num_train_epochs=epochs,
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


# 4. Train GPT-based model
def train_gpt(llm_name, train_texts, test_texts, epochs, fine_tune=False, output_dir='./model/gpt_lense'):
    if fine_tune and os.path.exists(output_dir):
        model, tokenizer = load_model_and_tokenizer(output_dir, AutoModelForCausalLM)
        print("GPTLense is fine-tuned on GPTLense again")
    else:
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
        num_train_epochs=epochs,
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


# 5. Fake News Detection Model
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
