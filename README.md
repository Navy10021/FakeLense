![header](https://capsule-render.vercel.app/api?type=waving&color=0:8B0000,50:FF0000,100:DC143C&height=300&section=header&text=FakeLense&fontColor=8B0000&fontSize=110&fontAlignY=40&fontAlign=50&animation=fadeIn&fontStyle=stroke)


<div align="center">
  
# 📰🔍 FakeLense: Leveraging a Hybrid BERT-GPT Model for Robust Detection of Disinformation

</div>


## 🎓 Introduction

ake news, defined as intentionally fabricated information designed to mislead, manipulate, or gain attention for political, economic, or other malicious purposes, has become one of the most pressing challenges of the digital age. Its rapid proliferation, fueled by the ubiquitous nature of social media (SNS) and the advancements in artificial intelligence (AI), has caused increasingly severe societal impacts. From undermining democratic processes to eroding trust in credible institutions, the consequences of fake news are far-reaching and multifaceted.

The advent of generative AI technologies has further exacerbated this issue, enabling the creation of highly realistic yet deceptive content that is increasingly difficult to distinguish from genuine information. Fake news creators have become more sophisticated, employing advanced AI tools to automate the generation of misleading narratives while concealing their identities. This has not only heightened the volume of disinformation but has also deepened social divisions, fueled polarization, and destabilized public trust.

Recognizing the significant security threats posed by false and manipulated information, governments and organizations worldwide are intensifying efforts to address these challenges. New frameworks, strategies, and specialized entities are being established to combat the spread of disinformation, protect democratic processes, and preserve societal cohesion.

**FakeLense** is an innovative response to this global challenge. Leveraging state-of-the-art natural language processing (NLP) technology, this project aims to detect and prevent the dissemination of fake news with unprecedented accuracy. By combining advanced machine learning models, real-time detection capabilities, and a hybrid approach using BERT and GPT technologies, **FakeLense** is designed not only to identify disinformation effectively but also to adapt dynamically to evolving patterns of malicious content. This cutting-edge tool aspires to empower governments, organizations, and individuals in their fight against disinformation, safeguarding the integrity of information and fostering a more informed and cohesive society.

## 📑 Project Overview

**FakeLense** is an advanced NLP-powered tool meticulously crafted to automatically detect and prevent the proliferation of fake news and disinformation. By leveraging a hybrid detection system that integrates the text comprehension prowess of **BERT** (Bidirectional Encoder Representations from Transformers) and the generative capabilities of **GPT** (Generative Pre-trained Transformer) Large Language Models, **FakeLense** is designed to achieve exceptional accuracy in identifying and countering disinformation. This **hybrid approach** not only ensures precision in classification but also enhances adaptability to evolving patterns of malicious content.

To train these models effectively, a comprehensive dataset of **63,678 real and fake news texts** was crawled and meticulously preprocessed. This dataset serves as a robust foundation, enabling **FakeLense** to consistently distinguish between genuine content and disinformation across diverse contexts.

An integral part of the **FakeLense** pipeline is the **Advanced NLP-Based Filtering Phase**, a sophisticated preprocessing mechanism that optimizes the detection system. This filtering phase employs cutting-edge techniques to expand, refine, and filter input data, ensuring only the most relevant and meaningful content is passed to the detection models. The filtering process includes: 1) **Keyword Expansion** using WordNet, Word2Vec, Sentence Transformers, and GPT. 2) **TF-IDF Filtering** for relevance scoring. 3) **Combined Filtering** to integrate keyword matching and TF-IDF scoring for maximum precision.
A dual-layer filtering mechanism that integrates keyword-based matching with TF-IDF relevance scoring to maximize precision.
**FakeLense** is not just a detection tool—it is a dynamic and intelligent solution designed to evolve alongside the challenges posed by modern disinformation campaigns. Its real-time capabilities, combined with a focus on adaptability and accuracy, make it a crucial resource for governments, organizations, and individuals striving to combat fake news effectively. By preprocessing, analyzing, and detecting disinformation with state-of-the-art methods, **FakeLense** contributes to safeguarding public trust, promoting informed decision-making, and mitigating the societal harms caused by disinformation.


![overall_pipeline](https://github.com/user-attachments/assets/2181f105-a6fe-49cb-8c90-97597a24e146)



### Key Features
  - **Hybrid Model Approach:** **FakeLense** combines BERT-based and GPT-based models to enhance detection accuracy.
  - **Focus on Disinformation:** The project targets the detection and prevention of disinformation—factually incorrect and maliciously intended content.
  - **Advanced Filtering:** By incorporating an advanced filtering phase, FakeLense not only improves detection accuracy but also strengthens its capability to address real-world challenges in combating disinformation.
  - **Real-time Detection:** **FakeLense** can be integrated into content platforms to monitor and flag potential fake news in real-time, preventing dissemination.
  - **Mitigating Social Harm:** By effectively blocking fake news, **FakeLense** aims to reduce unnecessary social conflicts, prevent the polarization of public opinion, and save national resources from being wasted on dealing with the consequences of misinformation.

### What is Fake News?

Fake news can be categorized into three main types:

  1. **DisInformation:** Information that is factually incorrect and maliciously intended (false O, malicious O).
  2. **Exaggerated Information:** Information that is factually correct but exaggerated with malicious intent (false X, malicious O).
  3. **MisInformation:** Information that is factually incorrect but not maliciously intended (false O, malicious X).

**FakeLense** focuses on detecting and blocking **Disinformation**—the most harmful type of fake news that misleads the public with incorrect data and malicious motives.

## 🛠️ Usage
Before you begin, ensure that you have Python 3.7 or higher installed. Install the required dependencies with the following command:
 ```bash
   pip install torch transformers scikit-learn pandas datasets
   ```
These dependencies include essential libraries for machine learning, natural language processing, and data handling.

### STEP 0. Clone the Repository
Start by cloning the repository to your local machine:
 ```bash
   git clone https://github.com/Navy10021/FakeLense.git
   cd FakeLense
   ```

### STEP 1. Prepare the Dataset
In this project, 63,678 real and fake news texts were crawled to train the **FakeLense** model. To prepare the dataset, run the ***'preprocessing.py'*** script. This script will automatically preprocess the text, label it, and split it into training and testing datasets with an 8:2 ratio.

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

### STEP 2. Training
To train both the BERT and GPT models, run the ***'train.py'*** script:
 ```bash
   python train.py
   ```
This script will:
  - Fine-tune the BERT-based model and save it in ./model/bert_lense.
  - Fine-tune the GPT-based model and save it in ./model/gpt_lense.

### STEP 3. Detection
After training, you can perform fake news detection by running the ***'detect.py'*** script:
 ```bash
   python detect.py
   ```
You can modify the 'test_cases' list in the script with your own examples for testing.
 ```python
   test_cases = [
    "In the wake of the recent election, residents of Amherst gathered at the local common...",
    "In a shocking twist, FBI Special Agent David Raynor, who was reportedly investigating a connection between Hillary Clinton...",
    ...
]
   ```
An example output is as follows:
 ```yaml
News 1: Real News Detected.
News 2: Fake News Detected.
News 3: Fake News Detected.
News 4: Real News Detected.
 ```

### STEP 4. Filtering
After training and detection, you can apply advanced filtering techniques to preprocess and refine input text for fake news detection. These filtering steps leverage NLP-based keyword expansion and TF-IDF scoring to ensure only relevant content is passed to the FakeLense detection pipeline.
Run the ***'filter.py'*** script to apply advanced filtering:
 ```bash
   python filter.py
   ```
The ***filter.py*** script includes:

#### 1. Keyword Expansion.
  - Expands the initial keyword set using: WordNet (semantic synonyms), Word2Vec (embedding-based similarity), Sentence Transformers (contextual similarity), GPT (generative keyword extension).

#### 2. TF-IDF Filtering.
Filters input text based on its relevance score using TF-IDF.

#### 3. Combined Filtering.
Combines **1) keyword-based** and **2) TF-IDF-based filters** to optimize input text for the detection phase.
Modify the ***test_texts*** list in ***'filter.py'*** to use your own examples:
 ```python
   test_cases = [
    "In the wake of the recent election, residents of Amherst gathered at the local common...",
    "In a shocking twist, FBI Special Agent David Raynor, who was reportedly investigating a connection between Hillary Clinton...",
    ...
]
   ```

An example output:
 ```bash
[PASS] Relevant text: Cyber attacks are becoming more frequent globally.
[PASS] Relevant text: The government plans to tackle fake news through AI systems.
[FILTERED] Irrelevant text: Unrelated text about cooking recipes.
   ```


## 🏋️‍♂️ Training Phase
### BERTLense: Train BERT-Based Model
BERT-based models can be fine-tuned using the ***'train_bert'*** function on pre-trained BERT-based LLMs. Here, you can build **BERTLense** by applying various BERT-based models. The default for LLMs is 'roberta-base'.
 ```python
   bert_trainer, bert_lense, bert_tokenizer = train_bert('microsoft/deberta-base', train_texts, train_labels, test_texts, test_labels)
   ```
### GPTLense: Train GPT-Based Model
GPT-based models can be fine-tuned using the ***'train_gpt'*** function on pre-trained GPT-based LLMs. This function allows you to build **GPTLense** by applying various GPT-based models. The default for LLMs is 'gpt2'.
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

## 🔬 Filtering Phase
The **Advanced NLP-Based Keyword Filtering** process preprocesses the input text to maximize the accuracy and relevance of the FakeLense detection model. The filtering process integrates the following key steps:
### 1. Keyword Expansion
The filtering pipeline begins by expanding the initial keyword set. This ensures a broader and more comprehensive detection of relevant content:
  - **WordNet Expansion**: Adds synonyms and semantically related terms.
  - **Word2Vec Expansion**: Identifies terms with high similarity in vector space.
  - **Sentence Transformers Expansion**: Finds contextually similar phrases and sentences.
  - **GPT-Based Expansion**: Generates new, relevant keywords using a generative language model.

### 2. TF-IDF Filtering
Using the expanded keywords as a base, a TF-IDF vectorizer is trained to compute relevance scores for input text. Texts with scores below the defined threshold are filtered out, ensuring only the most relevant content is passed to the FakeLense detection system.

### 3. Combined Filtering
The final phase combines keyword-based matching and TF-IDF relevance scoring to preprocess input text efficiently. Texts must match one or more expanded keywords and meet the TF-IDF threshold to pass.

### Benefits
  - **Accuracy Boost**: Ensures the detection models receive only high-quality and relevant data, leading to more accurate fake news classification.
  - **Adaptability**: The dynamic keyword expansion techniques allow the system to adapt to new trends and patterns in disinformation.
  - **Efficiency:** Reduces computational load by eliminating unnecessary or irrelevant inputs early in the pipeline.

## 📈 Fake News Detection Performance Evaluation Results
The experimental results demonstrated a **high detection accuracy of over 98%**, proving the tool's effectiveness in identifying fake news. **FakeLense** is expected to serve as an innovative "cognitive warfare" tool, capable of addressing misinformation across various channels and contributing to national interests.

![image](https://github.com/user-attachments/assets/eb230d6e-609a-4d88-b43f-8e3b5d8d0794)

## 👨‍💻 Contributors
- **Seoul National University Graduate School of Data Science (SNU GSDS)**
- Under the guidance of ***Navy Lee***

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.
