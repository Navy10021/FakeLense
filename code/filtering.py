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


# MAIN CODE
if __name__ == "__main__":
    # 1. Load Pretrained Models
    print(">> Loading models...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')        # Sentence-BERT model
    gpt_model = pipeline("text-generation", model="distilgpt2")     # GPT-2 model pipeline

    # 2. Initialize Base Keywords for Each Category
    base_keywords = {
        "Society": {"education", "crime", "accident", "poverty", "healthcare", "inequality"},
        "Politics": {"election", "president", "party", "policy", "diplomacy", "legislation"},
        "Economy": {"finance", "stock", "trade", "inflation", "recession", "unemployment"},
        "Issues": {"environment", "energy", "pandemic", "climate change", "biodiversity", "sustainability"},
        "Technology": {"artificial intelligence", "blockchain", "cybersecurity", "quantum computing", "5G", "automation"},
        "Sports": {"football", "basketball", "tournament", "championship"},
        "Science": {"astronomy", "biology", "physics", "chemistry"},
    }

    # 3. Configure GPT Expansion
    use_gpt = False  # Set to False to disable GPT-based keyword expansion

    # 4. Expand Keywords for Each Category
    print(">> Expanding keywords...")
    category_keywords = expand_category_keywords(base_keywords, sentence_model, gpt_model, use_gpt=use_gpt)

    # 5. Test News Data
    test_news = [
    {"text": "As global temperatures continue to rise, countries worldwide are facing increased pressure to adopt renewable energy sources. Recent climate reports emphasize the need for immediate action to reduce greenhouse gas emissions. Governments are implementing stricter policies, while companies invest in wind and solar power to meet sustainability goals. However, debates persist about balancing economic growth with environmental protection, as some regions still depend heavily on fossil fuels.",
     "category": "Issues"},
    {"text": "The race to develop cutting-edge artificial intelligence is intensifying as major tech companies announce breakthroughs in healthcare innovation. AI-powered diagnostics and personalized medicine are becoming a reality, with algorithms now capable of detecting diseases like cancer at earlier stages. Despite these advancements, experts warn of potential ethical concerns, including patient data privacy and the risk of biased algorithms in critical medical decisions.",
     "category": "Technology"},
    {"text": "Quantum computing is making headlines as scientists unveil a new system capable of solving complex problems exponentially faster than traditional computers. This breakthrough has significant implications for fields ranging from cryptography to pharmaceutical research. Companies like Google and IBM are leading the charge, investing billions in quantum research. However, challenges remain in scaling the technology and ensuring its practical applications for industry use.",
     "category": "Technology"},
    {"text": "The president has announced a comprehensive tax reform plan aimed at stimulating the economy and supporting small and medium-sized enterprises. Key features of the proposal include lower corporate tax rates and incentives for startups. While proponents argue these changes will spur innovation and job creation, critics warn that the reforms may increase the national deficit and place additional burdens on other sectors. The debate continues in Congress as both sides present their arguments.",
     "category": "Politics"},
    {"text": "Environmental activists are urging governments to take immediate action against deforestation in the Amazon, which has reached alarming levels in recent years. Satellite data shows significant forest loss, threatening biodiversity and accelerating climate change. Conservation groups are calling for stricter regulations, while local communities advocate for sustainable development models that balance economic needs with ecological preservation. Global attention is on policymakers to address this urgent issue.",
     "category": "Issues"},
    {"text": "Central banks worldwide are grappling with rising inflation rates, forcing them to rethink monetary policies. In a bid to stabilize economies, several nations have raised interest rates, but this has led to concerns about slowing economic growth. Analysts predict further rate hikes in the coming months, which could impact consumer spending and global trade. The ongoing geopolitical tensions and supply chain disruptions are also exacerbating the economic uncertainty.",
     "category": "Economy"},
    {"text": "Stock markets are experiencing volatile shifts as geopolitical tensions and fluctuating energy prices create uncertainty for investors. Major indices have shown sharp declines, reflecting concerns about global economic stability. Experts note that sectors like technology and energy are particularly vulnerable, while some investors are turning to safer assets like bonds and gold. Governments are closely monitoring the situation, with potential interventions to stabilize markets under consideration.",
     "category": "Economy"},
    {"text": "Healthcare professionals are emphasizing the importance of mental health awareness as National Mental Health Month begins. Campaigns across the country aim to reduce stigma and encourage individuals to seek help. Recent studies highlight the growing prevalence of anxiety and depression, particularly among young adults. Experts stress the need for increased funding in mental health services and the development of innovative therapies to address this public health crisis effectively.",
     "category": "Society"},
    {"text": "The FIFA World Cup has captivated audiences worldwide, breaking viewership records as millions tune in to watch their favorite teams compete. This year’s tournament has seen several dramatic matches, with underdog teams delivering surprising victories. Beyond the excitement, the event has sparked discussions on issues like player health, tournament logistics, and the environmental impact of hosting large-scale sports events. Fans eagerly await the next round of matches.",
    "category": "Sports"},
    {"text": "Astronomers have discovered an Earth-like exoplanet orbiting a star just 12 light-years away, sparking excitement in the scientific community. The planet, located in the habitable zone, has conditions that may support liquid water and potentially life. This finding is part of an ongoing effort to identify planets outside our solar system that could one day become destinations for interstellar exploration. Researchers are already planning follow-up observations using advanced telescopes.",
    "category": "Science"},
]


    # 6. Process and Visualize News by Category
    for category, keywords in category_keywords.items():
        print(f"\n>> Category: {category}")
        # Get news for the current category
        category_news = [news["text"] for news in test_news if news["category"] == category]
        if not category_news:  # Check if category_news is empty
            print(f"No news found for category: {category}")
            continue  # Skip this category if no news

        # Compute scores
        scores = [score_news_with_embeddings(news, keywords, sentence_model) for news in category_news]
        if not scores:  # Check if scores is empty
            print(f"No scores calculated for category: {category}")
            continue  # Skip this category if no scores

        # Optimize threshold
        threshold = optimize_threshold(scores, method='percentile', value=50)
        print(f"Optimized Threshold for {category}: {threshold:.2f}")

        # Visualize result
        visualize_scores(scores, threshold, category)

        # Display Filtering Results
        for news, score in zip(category_news, scores):
            status = "[PASS]" if score >= threshold else "[FILTERED]"
            print(f"{status} [Score: {score:.4f}] {news}")
