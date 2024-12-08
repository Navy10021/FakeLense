import nltk
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import wordnet
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
nltk.download('wordnet')  # WordNet
nltk.download('punkt')    # Tokenizer
nltk.download('punkt_tab')


# 1. WordNet Synonym Expansion
def expand_keywords_with_wordnet(keywords):
    """
    Use WordNet to expand keywords with synonyms.
    :param keywords: Initial set of keywords
    :return: Expanded set of keywords
    """
    expanded_keywords = set(keywords)
    for keyword in keywords:
        for synset in wordnet.synsets(keyword):                         # Retrieve WordNet synsets for the keyword
            for lemma in synset.lemmas():                               # Retrieve all lemmas (synonyms) in the synset
                expanded_keywords.add(lemma.name().replace("_", " "))   # Add synonyms to the set
    return expanded_keywords


# 2. Sentence Transformers Keyword Expansion
def expand_keywords_with_sentence_transformer(keywords, sentence_model):
    """
    Use Sentence-BERT to expand keywords by generating related phrases.
    :param keywords: Initial set of keywords
    :param sentence_model: Pretrained Sentence-BERT model
    :return: Expanded set of keywords
    """
    expanded_keywords = set(keywords)
    for keyword in keywords:
        # Generate embedding for the keyword
        keyword_embedding = sentence_model.encode(keyword, convert_to_tensor=True)
        # Generate related phrases
        related_phrases = [
            f"related to {keyword}",
            f"impact of {keyword}",
            f"challenges in {keyword}"
        ]
        # Compare similarity for each phrase
        for phrase in related_phrases:
            phrase_embedding = sentence_model.encode(phrase, convert_to_tensor=True)
            similarity = util.cos_sim(keyword_embedding, phrase_embedding).item()
            if similarity > 0.7:  # Add phrases with similarity > 0.7
                expanded_keywords.add(phrase)
    return expanded_keywords


# 3. GPT Keyword Expansion
def expand_keywords_with_gpt(keywords, gpt_model, max_length=50, num_return_sequences=1):
    """
    Use GPT model to generate related keywords and phrases.
    :param keywords: Initial set of keywords
    :param gpt_model: HuggingFace GPT model pipeline
    :param max_length: Maximum length of the generated text
    :param num_return_sequences: Number of generated sequences for each keyword
    :return: Expanded set of keywords
    """
    expanded_keywords = set(keywords)
    for keyword in keywords:
        prompt = f"Generate related keywords and phrases for '{keyword}':"
        try:
            responses = gpt_model(
                prompt,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                truncation=True,  # Explicit truncation
                pad_token_id=gpt_model.tokenizer.eos_token_id  # Explicit padding token ID
            )
            for response in responses:
                generated_text = response['generated_text']
                expanded_keywords.update(generated_text.split(", "))  # Add generated keywords
        except Exception as e:
            print(f"Error expanding keyword '{keyword}' with GPT: {e}")
    return expanded_keywords

# 4. Category Keyword Expansion
def expand_category_keywords(base_keywords, sentence_model, gpt_model, use_gpt=False):
    """
    Expand keywords for each category using WordNet, Sentence-BERT, and optionally GPT.
    :param base_keywords: Dictionary of initial keywords by category
    :param sentence_model: Pretrained Sentence-BERT model
    :param gpt_model: HuggingFace GPT model pipeline
    :param use_gpt: Boolean flag to enable/disable GPT-based keyword expansion
    :return: Dictionary of expanded keywords by category
    """
    category_keywords = {}
    for category, keywords in base_keywords.items():
        print(f">> Expanding keywords for category: {category}")
        expanded_keywords = expand_keywords_with_wordnet(keywords)  # WordNet expansion
        expanded_keywords = expand_keywords_with_sentence_transformer(expanded_keywords, sentence_model)  # S-BERT
        if use_gpt:  # Only expand with GPT if enabled
            expanded_keywords = expand_keywords_with_gpt(expanded_keywords, gpt_model)
        category_keywords[category] = expanded_keywords
    return category_keywords


# 5. Score News Using Sentence Embeddings
def score_news_with_embeddings(news_text, category_keywords, sentence_model, split_into_sentences=True):
    """
    Calculate similarity score between news text and category keywords using Sentence-BERT.
    Supports both whole-text embedding and sentence-level embedding approaches.
    
    :param news_text: Input news text
    :param category_keywords: Set of expanded keywords for the category
    :param sentence_model: Pretrained Sentence-BERT model
    :param split_into_sentences: If True, calculates scores at the sentence level; otherwise, whole-text level
    :return: Average similarity score
    """
    if split_into_sentences:
        # Split the news text into individual sentences
        sentences = nltk.sent_tokenize(news_text)
        # Compute embeddings for each sentence
        text_embeddings = [sentence_model.encode(sentence, convert_to_tensor=True) for sentence in sentences]
    else:
        # Compute a single embedding for the entire news text
        text_embeddings = [sentence_model.encode(news_text, convert_to_tensor=True)]

    # Compute embeddings for each keyword in the category
    keyword_embeddings = [sentence_model.encode(keyword, convert_to_tensor=True) for keyword in category_keywords]

    # Compute cosine similarity between text embeddings and keyword embeddings
    scores = [util.cos_sim(text_embedding, keyword_embedding).item()
              for text_embedding in text_embeddings
              for keyword_embedding in keyword_embeddings]

    return np.mean(scores)              # Return the average similarity score                                                              


# 6. Filter News by Score
def filter_news_with_score(news_text, category_keywords, sentence_model, threshold=0.5):
    """
    Filter news text based on similarity score with category keywords.
    :param news_text: Input news text
    :param category_keywords: Set of expanded keywords for the category
    :param sentence_model: Pretrained Sentence-BERT model
    :param threshold: Similarity threshold for filtering
    :return: Tuple (Pass/Fail, Similarity Score)
    """
    score = score_news_with_embeddings(news_text, category_keywords, sentence_model)
    return score >= threshold, score


# 7. Optimize Threshold
def optimize_threshold(scores, method='percentile', value=50):
    """
    Optimize the threshold for filtering based on score distribution.
    :param scores: List of similarity scores
    :param method: Method to calculate threshold ('percentile' or 'mean')
    :param value: Percentile value if method is 'percentile'
    :return: Optimized threshold
    """
    if method == 'percentile':
        return np.percentile(scores, value)     # Return the specified percentile
    elif method == 'mean':
        return np.mean(scores)                  # Return the mean of scores
    else:
        raise ValueError("Invalid method. Choose 'percentile' or 'mean'.")


# 8. Visualize Score Distribution
def visualize_scores(scores, threshold, category):
    """
    Visualize the score distribution for a category.
    :param scores: List of similarity scores
    :param threshold: Optimized threshold for filtering
    :param category: Category name
    """
    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=15, alpha=0.7, label='Scores')  # Histogram of scores
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.2f}')
    plt.title(f'Score Distribution for Category: {category}')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()
