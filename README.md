<div align="center">
  
# 📰🔍 FakeLense: A Hybrid BERT-GPT Approach for Robust Fake Content Detection

</div>


## 🎓 Introduction

Fake news refers to intentionally fabricated information spread to mislead, manipulate, or gain attention for political, economic, or other malicious purposes. The rapid proliferation of fake news, fueled by advancements in social media (SNS) and artificial intelligence (AI), has led to increasingly severe negative impacts on society. With the development of generative AI technologies, the risk of widespread distribution of fake news that is difficult to distinguish from genuine information has amplified. Moreover, fake news creators are becoming more sophisticated, concealing their identities and engaging in activities exacerbate social division and polarization.

In response to the significant security threats posed by false and manipulated information, governments worldwide are establishing new organizations dedicated to combating these issues. This project, **FakeLense**, aims to contribute to these efforts by developing an advanced tool for detecting fake news using cutting-edge natural language processing (NLP) technology.

## 📑 Project Overview

**FakeLense** is an AI-driven tool specifically designed to automatically detect and prevent the spread of fake news. By leveraging a hybrid detection system that combines the strengths of **BERT**(Bidirectional Encoder Representations from Transformers) and **GPT** (Generative Pre-trained Transformer) **Large Language Models**, FakeLense aims to achieve high accuracy in identifying and neutralizing fake news.

![overall_pipeline](https://github.com/user-attachments/assets/2181f105-a6fe-49cb-8c90-97597a24e146)



### Key Features
  - **Hybrid Model Approach:** **FakeLense** combines BERT-based and GPT-based models to enhance detection accuracy.
  - **Focus on False Information:** The project targets the detection and prevention of false information—content that is both factually incorrect and maliciously intended.
  - **Real-time Detection:** **FakeLense** can be integrated into content platforms to monitor and flag potential fake news in real-time, preventing its dissemination.
  - **Mitigating Social Harm:** By effectively blocking fake news, **FakeLense** aims to reduce unnecessary social conflicts, prevent the polarization of public opinion, and save national resources from being wasted on dealing with the consequences of misinformation.

### What is Fake News?

Fake news can be categorized into three main types:

  1. **False Information:** Information that is factually incorrect and maliciously intended (false O, malicious O).
  2. **Exaggerated Information:** Information that is factually correct but exaggerated with malicious intent (false X, malicious O).
  3. **Distorted Information:** Information that is factually incorrect but not maliciously intended (false O, malicious X).

**FakeLense** focuses on detecting and blocking **false information**—the most harmful type of fake news that misleads the public with incorrect data and malicious motives.


## 🛠️ Usage
Before you begin, ensure that you have Python 3.7 or higher installed. Install the required dependencies with the following command:
 ```bash
   pip install torch transformers scikit-learn pandas datasets
   ```
These dependencies include essential libraries for machine learning, natural language processing, and data handling.

### STEP 1. Clone the Repository
Start by cloning the repository to your local machine:
 ```bash
   git clone https://github.com/Navy10021/FakeLense.git
   cd FakeLense
   ```

### STEP 2. Prepare the Dataset
To prepare the dataset, you need to run the ***preprocessing.py*** script. This script will automatically preprocess the text, label it, and split it into a training and testing dataset with an 8:2 ratio.

Run the following command:
 ```bash
   python preprocessing.py
   ```
After running this script, you should have two files in the ./data/ folder:
  - train.csv: Training data
  - test.csv: Testing data

Each CSV file will have the following columns:
  - text: The text of the news article.
  - title : The title of the news article
  - target: The label (0 for real, 1 for fake).

### STEP 3. Training
To train both the BERT and GPT models, run the ***train.py*** script:
 ```bash
   python train.py
   ```
This script will:
  - Fine-tune the BERT-based model and save it in ./model/bert_lense.
  - Fine-tune the GPT-based model and save it in ./model/gpt_lense.

### STEP 4. Detection
After training, you can perform fake news detection by running the ***detect.py*** script:
 ```bash
   python detect.py
   ```
You can modify the test_cases list in the script with your own examples for testing.

### STEP 5. Sample Test Cases
The ***detect.py*** script includes several test cases. You can customize them with your examples:
 ```python
   test_cases = [
    "Global Leaders Gather for Climate Summit...",
    "Breaking News: The moon is made of cheese...",
    ...
]
   ```
An example output is as follows:
 ```yaml
News 1: Real News Detected.
News 2: Real News Detected.
News 3: Fake News Detected.
News 4: Fake News Detected.
 ```

## 🏋️‍♂️ Training Phase
### BERTLense: Train BERT-Based Model
BERT-based models can be fine-tuned using the ***train_bert*** function on pre-trained BERT-based LLMs. Here, you can build **BERTLense** by applying various BERT-based models. The default for LLMs is 'roberta-base'.
 ```python
   bert_trainer, bert_lense, bert_tokenizer = train_bert('microsoft/deberta-base', train_texts, train_labels, test_texts, test_labels)
   ```
### GPTLense: Train GPT-Based Model
GPT-based models can be fine-tuned using the ***train_gpt*** function on pre-trained GPT-based LLMs. This function allows you to build **GPTLense** by applying various GPT-based models. The default for LLMs is 'gpt2'.
 ```python
   gpt_trainer, gpt_lense, gpt_tokenizer = train_gpt('EleutherAI/gpt-neo-125M', train_texts, test_texts)
   ```
Both trained models will be saved in the ./model/ directory.

## 🕵️‍♂️ Detection Phase
The main feature of this code is its implementation, which focuses on enhancing the accuracy of fake news detection by combining the strengths of BERT and GPT. BERT excels at text classification, while GPT provides an additional verification step through its text generation capabilities. Specifically, the ***FakeLense function*** synthesizes the results of both models: it identifies fake news if BERT classifies the text as such or if the similarity between the generated text by GPT and the original text is low. This process is used as a strategy to improve the accuracy of fake news detection.
 ```python
   def FakeLense(text, bert_model, bert_tokenizer, gpt_model, gpt_tokenizer, similarity_threshold=0.8):
    ...
   ```

## 👨‍💻 Contributors
- **Seoul National University Graduate School of Data Science (SNU GSDS)**
- Under the guidance of ***Navy Lee***

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.
