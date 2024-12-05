from nlp_filter import * 


if __name__ == "__main__":
    # 0. Load Pre-trained models
    print(">> Loading models...")
    word2vec_model = KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    gpt_model = pipeline("text-generation", model="distilgpt2")

    # 1. Initial base keywords setting
    base_keywords = {"cyber", "attack", "fake news", "disinformation", "security", "threat"}

    # 2. Keywords expansion
    print(">> Expanding keywords...")
    keywords = expand_keywords_with_wordnet(base_keywords)
    for keyword in base_keywords:
        keywords.update(expand_keywords_with_word2vec(keyword, word2vec_model))
    keywords = expand_keywords_with_sentence_transformer(keywords, sentence_model)
    keywords = expand_keywords_with_gpt(keywords, gpt_model)
    # Remove duplicates in final keywords
    keywords = set(keywords)
    print("Final expanded keywords:", sorted(keywords)) 

    # 3. TF-IDF Train
    print(">> Training TF-IDF...")
    # User settings based on real news
    corpus = [
        "Cyber attacks have become increasingly sophisticated, targeting critical infrastructure and government systems.",
        "Fake news campaigns on social media platforms are influencing public opinion and undermining democratic processes.",
        "Disinformation campaigns are being utilized by state and non-state actors",
    ]
    tfidf_vectorizer, tfidf_matrix = train_tfidf(corpus)


    # 4. real time test data input
    test_texts = [
        "Cyber attacks are a growing threat to security.",
        "Fake news is being used as a weapon of disinformation.",
        "This message is irrelevant to the topic."
    ]

    print("\nFiltering test texts...")
    for idx, text in enumerate(test_texts, 1):
        result = "[PASS]" if filter_text(text, keywords, tfidf_vectorizer, use_tfidf=True) else "[FILTERED]"
        print(f"{result} Test {idx}: {text}")
