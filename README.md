![header](https://capsule-render.vercel.app/api?type=waving&color=0:8B0000,50:FF0000,100:DC143C&height=300&section=header&text=FakeLense&fontColor=8B0000&fontSize=110&fontAlignY=40&fontAlign=50&animation=fadeIn&fontStyle=stroke)


<div align="center">
  
# 📰🔍 FakeLense: Leveraging a Hybrid BERT-GPT Model for Robust Detection of Disinformation

</div>

## 🎓 Introduction

Fake news, intentionally fabricated to mislead or manipulate for political, economic, or malicious purposes, has become a major challenge in the digital age. The rise of social media and advancements in AI have accelerated its spread, leading to significant societal impacts, including eroding trust in institutions and undermining democratic processes.

Generative AI has worsened the problem by enabling the creation of highly realistic deceptive content, making disinformation harder to detect. Sophisticated fake news creators now use advanced AI tools to amplify disinformation, deepen social divisions, and destabilize public trust.

To address these threats, governments and organizations are adopting new strategies and frameworks to combat disinformation and protect societal integrity.

**FakeLense** is an innovative solution to this global issue. Powered by cutting-edge NLP technologies like BERT and GPT, it offers real-time, highly accurate detection of fake news. This dynamic tool empowers individuals and organizations to combat disinformation, safeguard information integrity, and foster a more informed society.


## 📑 Project Overview

**FakeLense** is an advanced NLP-powered tool designed to detect and prevent the proliferation of fake news and disinformation. By combining the text comprehension capabilities of **BERT** and the generative strengths of **GPT**, it achieves exceptional accuracy in identifying disinformation while adapting dynamically to evolving malicious content.

The system is trained on a comprehensive dataset of 63,678 real and fake news texts, serving as a robust foundation for reliable detection across diverse contexts. Additionally, **FakeLense** incorporates real-time news monitoring and filtering, enabling continuous updates with high-quality data. This ensures the detection models remain effective and responsive to the ever-changing landscape of disinformation.

Another critical feature of **FakeLense** is its advanced **NLP-based filtering** Phase, which refines input data through techniques such as keyword expansion, semantic analysis, and TF-IDF-based precision filtering. By leveraging tools like WordNet, Word2Vec, Sentence Transformers, and GPT, it ensures only the most relevant and meaningful content is passed to the detection models, enhancing overall system performance.

**FakeLense** is a dynamic and adaptable solution built to counter modern disinformation campaigns effectively. Its real-time detection capabilities, combined with advanced preprocessing methods, empower governments, organizations, and individuals to combat fake news, safeguard public trust, and mitigate the societal harm caused by disinformation.


![image](https://github.com/user-attachments/assets/e0944223-5624-42ee-9b45-9ffc8d3c75fc)




### Key Features
  - **Hybrid Model Approach.**  'FakeLense' combines BERT-based and GPT-based models to enhance detection accuracy.
  - **Focus on Disinformation.**  The project targets the detection and prevention of disinformation—factually incorrect and maliciously intended content.
  - **Advanced Filtering.**  By incorporating an advanced filtering phase, 'FakeLense' not only improves detection accuracy but also strengthens its capability to address real-world challenges in combating disinformation.
  - **Real-time Detection.**  'FakeLense' can be integrated into content platforms to monitor and flag potential fake news in real-time, preventing dissemination.
  - **Mitigating Social Harm.**  By effectively blocking fake news, 'FakeLense' aims to reduce unnecessary social conflicts, prevent the polarization of public opinion, and save national resources from being wasted on dealing with the consequences of misinformation.

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


### STEP @. Filtering (Optional but Recommended)

Filtering is an optional but highly recommended step that can be applied both during the data construction phase and the detection pipeline. By preprocessing and refining the input text, you can improve the quality and relevance of data used for fake news detection. This ensures that only meaningful content is passed forward, enhancing the overall performance of the system.
  - During Data Construction: Apply filtering to preprocess raw data, ensuring a high-quality dataset for training and testing.
  - In the Detection Pipeline: Refine real-time input data before passing it to the FakeLense detection model.
By integrating filtering into both stages, you can maximize the accuracy and efficiency of the entire workflow.
 
 Run the **filtering.py** script to apply advanced NLP-based filtering:
 ```bash
   python filtering.py
   ```

Modify the ***test_texts*** list in **'filtering.py'** to use your own examples:

 ```python
   test_cases = [
    "In the wake of the recent election, residents of Amherst gathered at the local common...",
    "In a shocking twist, FBI Special Agent David Raynor, who was reportedly investigating a connection between Hillary Clinton...",
    ...
]
   ```

An example output:
 ```bash
[PASS] Relevant text: In the wake of the recent election, residents of Amherst gathered at the local common...
[PASS] Relevant text: In a shocking twist, FBI Special Agent David Raynor, who was reportedly investigating a connection between Hillary Clinton...
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
