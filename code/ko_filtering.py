from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import numpy as np
import matplotlib.pyplot as plt
from konlpy.tag import Okt  # 한국어 형태소 분석기
import kss                  # 한국어 문장 분리 라이브러리
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

# 형태소 분석기 초기화
okt = Okt()

# 1. 동의어 확장 함수
def expand_keywords_with_custom_list(keywords):
    """
    사용자 정의 동의어 확장 (한국어 기반).
    """
    synonym_dict = {
        "환경": {"기후 변화", "온실가스", "재생 에너지", "탄소 중립"},
        "정치": {"대통령", "총선", "정당", "의회"},
        "경제": {"물가", "금리", "주식시장", "무역"},
        # 필요한 카테고리에 따라 추가
    }
    expanded_keywords = set(keywords)
    for keyword in keywords:
        if keyword in synonym_dict:
            expanded_keywords.update(synonym_dict[keyword])
    return expanded_keywords

# 2. Sentence Transformers 확장
def expand_keywords_with_sentence_transformer(keywords, sentence_model):
    expanded_keywords = set(keywords)
    for keyword in keywords:
        keyword_embedding = sentence_model.encode(keyword, convert_to_tensor=True)
        related_phrases = [
            f"{keyword} 관련 이슈",
            f"{keyword}의 영향",
            f"{keyword} 해결 방안",
        ]
        for phrase in related_phrases:
            phrase_embedding = sentence_model.encode(phrase, convert_to_tensor=True)
            similarity = util.cos_sim(keyword_embedding, phrase_embedding).item()
            if similarity > 0.7:
                expanded_keywords.add(phrase)
    return expanded_keywords

# 3. 뉴스 텍스트 점수 계산
def score_news_with_embeddings(news_text, category_keywords, sentence_model, split_into_sentences=True):
    if split_into_sentences:
        sentences = kss.split_sentences(news_text)
        text_embeddings = [sentence_model.encode(sentence, convert_to_tensor=True) for sentence in sentences]
    else:
        text_embeddings = [sentence_model.encode(news_text, convert_to_tensor=True)]

    keyword_embeddings = [sentence_model.encode(keyword, convert_to_tensor=True) for keyword in category_keywords]

    scores = [util.cos_sim(text_embedding, keyword_embedding).item()
              for text_embedding in text_embeddings
              for keyword_embedding in keyword_embeddings]

    return np.mean(scores)

# 4. 필터링 및 시각화
def visualize_scores(scores, threshold, category):
    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=15, alpha=0.7, label='점수 분포')
    plt.axvline(threshold, color='red', linestyle='--', label=f'임계값: {threshold:.2f}')
    plt.title(f'카테고리: {category}의 점수 분포')
    plt.xlabel('점수')
    plt.ylabel('빈도수')
    plt.legend()
    plt.grid(True)
    plt.show()

# 5. 임계값 최적화
def optimize_threshold(scores, method='percentile', value=50):
    """
    점수 분포에 기반하여 필터링 임계값을 최적화합니다.
    :param scores: 점수 리스트
    :param method: 임계값 계산 방법 ('percentile' 또는 'mean')
    :param value: 'percentile' 선택 시 백분위수 값
    :return: 최적화된 임계값
    """
    if method == 'percentile':
        return np.percentile(scores, value)  # 지정한 백분위수 반환
    elif method == 'mean':
        return np.mean(scores)  # 점수 평균 반환
    else:
        raise ValueError("method는 'percentile' 또는 'mean'이어야 합니다.")

# 메인 코드
if __name__ == "__main__":
    # 1. 모델 로드
    print(">> 모델 로드 중...")
    sentence_model = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')  # 한국어 지원 모델
    gpt_model = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")  # GPT 모델 예시

    # 2. 카테고리 키워드 정의
    base_keywords = {
    "사회": {"교육", "범죄", "사고", "건강", "복지", "실업", "노동", "저출산", "고령화"},
    "정치": {"대통령", "선거", "정당", "정책", "국회", "외교", "행정", "입법", "사법"},
    "경제": {"금리", "물가", "무역", "주식", "환율", "소비", "투자", "재정", "GDP"},
    "기술": {"인공지능", "블록체인", "사이버 보안", "양자 컴퓨팅", "5G", "자동화", "IoT", "클라우드"},
    "환경": {"기후 변화", "탄소 중립", "재생 에너지", "온실가스", "수질 오염", "미세먼지", "환경 보호"},
    "스포츠": {"축구", "농구", "야구", "올림픽", "월드컵", "선수", "경기", "체육", "리그"},
    "과학": {"우주", "천문학", "생물학", "물리학", "화학", "지구과학", "기술 개발", "연구"},
}

    # 3. 키워드 확장
    print(">> 키워드 확장 중...")
    category_keywords = {category: expand_keywords_with_custom_list(keywords) for category, keywords in base_keywords.items()}

    # 4. 테스트 뉴스 데이터
    test_news = [
    {
        "text": "정부는 기후 변화 대응을 위해 새로운 재생 에너지 정책을 발표했습니다. 이번 정책은 태양광과 풍력 에너지의 보급을 확대하고, 탄소 배출을 줄이기 위해 다양한 산업의 규제를 강화하는 내용을 포함하고 있습니다. 정부 관계자는 이 정책이 2030년까지 탄소 중립 목표를 달성하는 데 중요한 역할을 할 것이라고 밝혔습니다.",
        "category": "환경"
    },
    {
        "text": "대통령 선거가 점점 치열해지고 있습니다. 주요 후보들은 경제 정책, 복지 정책, 외교 전략 등을 중심으로 격렬한 토론을 벌이고 있습니다. 특히, 유권자들은 실업 문제와 물가 상승에 대한 해결 방안에 주목하고 있습니다. 한편, 여론 조사에서는 후보 간의 지지율 격차가 점차 좁아지고 있어 선거 결과에 대한 예측이 어려운 상황입니다.",
        "category": "정치"
    },
    {
        "text": "축구 월드컵에서 예상치 못한 결과가 이어지며 전 세계 팬들의 관심이 쏠리고 있습니다. 몇몇 강팀이 조기 탈락의 위기에 처한 반면, 신흥 강국들이 새로운 강자로 떠오르고 있습니다. 경기장 안팎에서는 선수들의 부상 문제와 심판 판정 논란이 계속되고 있지만, 팬들은 여전히 열광적인 응원을 보내고 있습니다.",
        "category": "스포츠"
    },
    {
        "text": "양자 컴퓨팅 기술이 최근 혁신적인 발전을 이루며 주목받고 있습니다. 과학자들은 이 기술이 기존 암호화 방식을 무력화할 수 있을 정도로 강력한 계산 능력을 제공한다고 평가하고 있습니다. 기업들은 양자 컴퓨팅을 활용한 새로운 보안 솔루션을 개발하고 있으며, 관련 연구와 투자가 급격히 증가하고 있습니다.",
        "category": "기술"
    },
    {
        "text": "세계 경제가 금리 인상과 환율 변동으로 큰 혼란을 겪고 있습니다. 중앙은행들은 인플레이션 억제를 위해 금리를 연이어 인상하고 있지만, 이로 인해 소비와 투자가 위축되고 있습니다. 전문가들은 이러한 경제적 불확실성이 국제 무역에도 악영향을 미칠 가능성이 높다고 경고하고 있습니다.",
        "category": "경제"
    },
    {
        "text": "범죄율 증가로 인해 지역 사회가 큰 위협을 받고 있습니다. 특히, 최근 강력 범죄 사건이 연이어 발생하면서 주민들은 안전 대책 마련을 촉구하고 있습니다. 지역 경찰은 순찰을 강화하고 범죄 예방 프로그램을 확대하기로 했지만, 일부 주민들은 여전히 불안을 호소하고 있습니다.",
        "category": "사회"
    },
    {
        "text": "천문학자들이 새로운 외계 행성을 발견하며 우주 탐사의 새로운 장을 열고 있습니다. 이번에 발견된 행성은 지구와 유사한 환경을 가진 것으로 알려졌으며, 물이 존재할 가능성도 제기되고 있습니다. 연구팀은 더 발전된 망원경을 통해 해당 행성을 정밀 관찰할 계획이며, 이는 향후 우주 탐사 기술 발전에도 기여할 것으로 보입니다.",
        "category": "과학"
    },
    {
        "text": "미세먼지 농도가 심각한 수준에 도달하며 시민들의 건강이 위협받고 있습니다. 정부는 대기 질 개선을 위해 차량 운행 제한과 공장 배출 가스 규제를 강화하기로 했습니다. 전문가들은 장기적으로 친환경 에너지 전환이 필수적이라고 강조하며, 국민들에게 마스크 착용과 외출 자제를 권고하고 있습니다.",
        "category": "환경"
    },
    {
        "text": "정부는 교육 격차를 해소하기 위해 새로운 온라인 학습 플랫폼을 출시했습니다. 이번 플랫폼은 농어촌 지역 학생들도 양질의 교육 콘텐츠에 접근할 수 있도록 설계되었습니다. 전문가들은 이를 통해 학생들의 학업 성취도가 향상될 것으로 기대하고 있지만, 기술 격차 문제에 대한 해결 방안도 필요하다고 지적하고 있습니다.",
        "category": "사회"
    },
    {
        "text": "올림픽 개최 도시가 발표되며 스포츠 팬들의 기대가 높아지고 있습니다. 이번 올림픽은 친환경과 지속 가능성을 중심으로 기획될 예정이며, 선수들과 관중들에게 특별한 경험을 제공할 것으로 보입니다. 그러나 일부에서는 막대한 예산 문제와 환경 파괴 가능성에 대한 우려의 목소리도 나오고 있습니다.",
        "category": "스포츠"
    },
]

    # 5. 뉴스 필터링 및 시각화
    for category, keywords in category_keywords.items():
        print(f"\n>> 카테고리: {category}")
        category_news = [news["text"] for news in test_news if news["category"] == category]
        if not category_news:
            print(f"{category} 카테고리에 해당하는 뉴스가 없습니다.")
            continue

        # 점수 계산
        scores = [score_news_with_embeddings(news, keywords, sentence_model) for news in category_news]
        if not scores:
            print(f"{category} 카테고리에 점수가 계산되지 않았습니다.")
            continue

        # 임계값 최적화
        threshold = optimize_threshold(scores, method='percentile', value=50)  # 50th Percentile 기준
        print(f"{category} 임계값: {threshold:.2f}")

        # 점수 분포 시각화
        visualize_scores(scores, threshold, category)

        # 필터링 결과 출력
        for news, score in zip(category_news, scores):
            status = "[PASS]" if score >= threshold else "[FILTERED]"
            print(f"{status} [점수: {score:.4f}] {news}")
