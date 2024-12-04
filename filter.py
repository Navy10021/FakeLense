import nltk
nltk.download('wordnet')
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from gensim.models.keyedvectors import KeyedVectors
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 1. 초기 키워드 설정
# 초기 키워드 리스트를 설정합니다.
base_keywords = {"cyber", "attack", "fake news", "disinformation", "security", "threat"}

# 2. WordNet 기반 동의어 확장
def expand_keywords_with_wordnet(keywords):
    """
    WordNet을 활용해 초기 키워드의 동의어를 확장합니다.
    :param keywords: 초기 키워드 집합
    :return: 확장된 키워드 집합
    """
    expanded_keywords = set(keywords)  # 확장된 키워드 저장
    for keyword in keywords:
        # WordNet의 Synset에서 동의어 추출
        for synset in wordnet.synsets(keyword):
            for lemma in synset.lemmas():
                expanded_keywords.add(lemma.name().replace("_", " "))  # 언더스코어(_) 제거
    return expanded_keywords

# 3. Word Embedding 기반 확장 (Word2Vec)
def expand_keywords_with_word2vec(keyword, model, top_n=5):
    """
    Word2Vec 모델을 사용하여 키워드와 유사한 단어를 확장합니다.
    :param keyword: 단일 키워드
    :param model: Word2Vec 모델
    :param top_n: 유사 단어 개수
    :return: 유사 단어 집합
    """
    try:
        # Word2Vec으로 유사 단어를 가져옴
        similar_words = model.most_similar(keyword, topn=top_n)
        return {w[0] for w in similar_words}
    except KeyError:
        # 키워드가 모델에 없는 경우 빈 집합 반환
        return set()

# 4. Sentence Transformers 기반 의미 확장
def expand_keywords_with_sentence_transformer(keywords, sentence_model, top_n=3):
    """
    Sentence Transformers를 사용해 키워드의 의미를 확장합니다.
    :param keywords: 초기 키워드 집합
    :param sentence_model: Sentence Transformers 모델
    :param top_n: 관련 문구 개수
    :return: 확장된 키워드 집합
    """
    expanded_keywords = set(keywords)
    for keyword in keywords:
        # 키워드의 벡터 임베딩 생성
        keyword_embedding = sentence_model.encode(keyword, convert_to_tensor=True)
        # 키워드와 관련된 문장을 생성
        related_phrases = [
            f"related to {keyword}",
            f"impact of {keyword} on cybersecurity",
            f"challenges in {keyword}"
        ]
        # 문장별로 유사도를 계산
        for phrase in related_phrases:
            phrase_embedding = sentence_model.encode(phrase, convert_to_tensor=True)
            similarity = util.cos_sim(keyword_embedding, phrase_embedding).item()
            # 유사도가 임계값(0.7) 이상인 문구만 추가
            if similarity > 0.7:
                expanded_keywords.add(phrase)
    return expanded_keywords

# 5. Transformer 기반 키워드 확장 (GPT)
def expand_keywords_with_gpt(keywords, max_length=50, num_return_sequences=1):
    """
    GPT 모델을 사용해 키워드를 확장합니다.
    :param keywords: 초기 키워드 리스트
    :param max_length: 생성되는 텍스트의 최대 길이
    :param num_return_sequences: 반환할 결과물 수
    :return: 확장된 키워드 집합
    """
    expanded_keywords = set(keywords)
    
    for keyword in keywords:
        prompt = f"Generate related keywords and phrases for '{keyword}' in cybersecurity:"
        try:
            # 텍스트 생성
            responses = gpt_model(
                prompt,
                max_length=max_length,  # 생성되는 텍스트 길이 제한
                num_return_sequences=num_return_sequences,  # 반환할 결과물 수
                truncation=True,  # 텍스트를 트렁케이션
                pad_token_id=gpt_model.tokenizer.eos_token_id  # 패딩 토큰 명시
            )
            
            # 생성된 결과를 키워드로 추가
            for response in responses:
                generated_text = response['generated_text']
                expanded_keywords.update(generated_text.split(", "))
        
        except Exception as e:
            print(f"Error generating keywords for '{keyword}': {e}")
    
    return expanded_keywords

# 6. TF-IDF 기반 필터링
def train_tfidf(corpus):
    """
    TF-IDF 벡터라이저를 학습합니다.
    :param corpus: 학습용 코퍼스 (텍스트 리스트)
    :return: TF-IDF 벡터라이저와 학습된 TF-IDF 행렬
    """
    vectorizer = TfidfVectorizer(max_features=1000)  # 최대 1000개의 특징 사용
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix

def tfidf_filtering(text, tfidf_vectorizer, threshold=0.1):
    """
    TF-IDF 점수를 기반으로 텍스트를 필터링합니다.
    :param text: 입력 텍스트
    :param tfidf_vectorizer: 학습된 TF-IDF 벡터라이저
    :param threshold: 필터링 임계값
    :return: 필터링 여부 (True/False)
    """
    tfidf_scores = tfidf_vectorizer.transform([text]).toarray()
    max_score = np.max(tfidf_scores)  # TF-IDF 점수 중 최대값
    return max_score >= threshold

# 7. 텍스트 필터링 (통합)
def filter_text(text, expanded_keywords, tfidf_vectorizer=None, use_tfidf=False):
    """
    키워드 및 TF-IDF를 조합하여 텍스트를 필터링합니다.
    :param text: 입력 텍스트
    :param expanded_keywords: 확장된 키워드 집합
    :param tfidf_vectorizer: 학습된 TF-IDF 벡터라이저 (선택 사항)
    :param use_tfidf: TF-IDF 필터링 사용 여부
    :return: 필터링 여부 (True/False)
    """
    # 키워드 필터링
    tokens = nltk.word_tokenize(text.lower())  # 입력 텍스트 토큰화
    keyword_match = any(keyword.lower() in tokens for keyword in expanded_keywords)

    # TF-IDF 필터링
    if use_tfidf and tfidf_vectorizer is not None:
        tfidf_match = tfidf_filtering(text, tfidf_vectorizer)
    else:
        tfidf_match = True

    return keyword_match and tfidf_match




if __name__ == "__main__":
    # 사전 학습된 모델 로드
    print(">> Loading models...")
    word2vec_model = KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    gpt_model = pipeline("text-generation", model="distilgpt2")

    # 1.키워드 확장
    print(">> Expanding keywords...")
    keywords = expand_keywords_with_wordnet(base_keywords)
    for keyword in base_keywords:
        keywords.update(expand_keywords_with_word2vec(keyword, word2vec_model))
    keywords = expand_keywords_with_sentence_transformer(keywords, sentence_model)
    keywords = expand_keywords_with_gpt(keywords, gpt_model)

    print("Final expanded keywords:", keywords)

    # 2.TF-IDF 학습
    print(">> Training TF-IDF...")
    corpus = [
        "cyber attack is a growing threat to security",
        "fake news is spreading rapidly",
        "disinformation campaigns are common in modern conflicts",
        "security threats require better strategies"
    ]
    tfidf_vectorizer, tfidf_matrix = train_tfidf(corpus)


    # 실시간 테스트 데이터 입력
    test_texts = [
        "Cyber attacks are a growing threat to security.",
        "Fake news is being used as a weapon of disinformation.",
        "This message is irrelevant to the topic."
    ]

    # 필터링 실행
    print("\nFiltering test texts...")
    for text in test_texts:
        if filter_text(text, keywords, tfidf_vectorizer, use_tfidf=True):
            print(f"[PASS] Relevant text: {text}")
        else:
            print(f"[FILTERED] Irrelevant text: {text}")
