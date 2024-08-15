<div align="center">
  
# 📰🔍 FakeLense: A Hybrid BERT-GPT Approach for Robust Fake Content Detection

</div>
## 🎓 Introduction

Fake news refers to intentionally fabricated information spread to mislead, manipulate, or gain attention for political, economic, or other malicious purposes. The rapid proliferation of fake news, fueled by advancements in social media (SNS) and artificial intelligence (AI), has led to increasingly severe negative impacts on society. With the development of generative AI technologies, the risk of widespread distribution of fake news that is difficult to distinguish from genuine information has amplified. Moreover, fake news creators are becoming more sophisticated, concealing their identities and engaging in activities exacerbate social division and polarization.

In response to the significant security threats posed by false and manipulated information, governments worldwide are establishing new organizations dedicated to combating these issues. This project, **FakeLense**, aims to contribute to these efforts by developing an advanced tool for detecting fake news using cutting-edge natural language processing (NLP) technology.

## 📑 Project Overview

**FakeLense** is an AI tool specifically designed to automatically detect and block the spread of fake news. By leveraging a hybrid detection system that combines the strengths of **BERT(Bidirectional Encoder Representations from Transformers)** and **GPT(Generative Pre-trained Transformer) LLMs**, FakeLense aims to achieve high accuracy in identifying and neutralizing fake news.

![overall_pipeline](https://github.com/user-attachments/assets/397d61d1-7033-405b-b986-dbf1c2b701b7)


### Key Features:
  - **Hybrid Model Approach:** FakeLense combines BERT-based and GPT-based models to enhance detection accuracy.
  - **Focus on False Information:** The project targets the detection and prevention of false information—content that is both factually incorrect and maliciously intended.
  - **Real-time Detection:** FakeLense can be integrated into content platforms to monitor and flag potential fake news in real-time, preventing its dissemination.
  - **Mitigating Social Harm:** By effectively blocking fake news, FakeLense aims to reduce unnecessary social conflicts, prevent the polarization of public opinion, and save national resources from being wasted on dealing with the consequences of misinformation.

### Classification of Fake News:

Fake news can be categorized into three main types:

  1. **False Information:** Information that is factually incorrect and maliciously intended (false O, malicious O).
  2. **Exaggerated Information:** Information that is factually correct but exaggerated with malicious intent (false X, malicious O).
  3. **Distorted Information:** Information that is factually incorrect but not maliciously intended (false O, malicious X).

*FakeLense** focuses on detecting and blocking **false information**—the most harmful type of fake news that misleads the public with incorrect data and malicious motives.


## 🛠️ Usage
Before you begin, ensure that you have Python 3.7 or higher installed. Install the required dependencies with the following command:
 ```bash
   pip install torch transformers scikit-learn pandas datasets
   ```
These dependencies include essential libraries for machine learning, natural language processing, and data handling.

### 1. Clone the Repository
Start by cloning the repository to your local machine:
 ```bash
   git clone https://github.com/Navy10021/FakeLense.git
   cd FakeLense
   ```

### 2. Prepare the Dataset
To prepare the dataset, you need to run the preprocessing.py script. This script will automatically preprocess the text, label it, and split it into a training and testing dataset with an 80:20 ratio. The processed data will then be saved as train.csv and test.csv in the data/ folder.

Run the following command:
 ```bash
   python preprocessing.py
   ```
After running this script, you should have two files:
  - train.csv: Training data
  - test.csv: Testing data
Each CSV file will have the following columns:
  - text: The text of the news article.
  - target: The label (0 for real, 1 for fake).

### 3. Training
To train both the BERT and GPT models, run the train.py script:
 ```bash
   python train.py
   ```
This script will:
  - Fine-tune the BERT model and save it in ./model/bert_lense.
  - Fine-tune the GPT model and save it in ./model/gpt_lense.

### 4. Detection
After training, you can perform fake news detection by running the detect.py script:
 ```bash
   python detection.py
   ```
You can modify the test_cases list in the script with your own examples for testing.

### Example Output
 ```yaml
News 1 : Real News Detected.
News 2 : Real News Detected.
News 3 : Fake News Detected.
News 4 : Fake News Detected.
 ```

## 🏋️‍♂️ Training
### Train BERT-Based Model
The BERT-based model can be fine-tuned using the train_bert function:
 ```python
   bert_trainer, bert_lense, bert_tokenizer = train_bert('microsoft/deberta-base', train_texts, train_labels, test_texts, test_labels)
   ```
### Train GPT-Based Model
Fine-tune the GPT model using the train_gpt function:
 ```python
   gpt_trainer, gpt_lense, gpt_tokenizer = train_gpt('EleutherAI/gpt-neo-125M', train_texts, test_texts)
   ```
Both models will be saved in the ./model/ directory.

## 🔍 Detection
The **'FakeLense'** function combines the outputs of the BERT and GPT models to determine whether the news is real or fake:
 ```python
   def FakeLense(text, bert_model, bert_tokenizer, gpt_model, gpt_tokenizer, similarity_threshold=0.8):
    ...
   ```

## 🧪 Sample Test Cases
The **detecttion.py** script includes several test cases. You can customize them with your examples:
 ```python
   test_cases = [
    "Global Leaders Gather for Climate Summit...",
    "Breaking News: The moon is made of cheese...",
    ...
]
   ```

## 👨‍💻 Contributors
- **Seoul National University Graduate School of Data Science (SNU GSDS)**
- Under the guidance of ***Navy Lee***

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.
