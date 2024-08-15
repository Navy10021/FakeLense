from fake_lense import * 

# Detection Phase
bert_lense, bert_tokenizer = load_model_and_tokenizer('./model/bert_lense', AutoModelForSequenceClassification)
gpt_lense, gpt_tokenizer = load_model_and_tokenizer('./model/gpt_lense', AutoModelForCausalLM)

test_texts = [
    # Truth News
    "Global Leaders Gather for Climate Summit: World leaders have convened in Paris for the annual climate summit, where they are expected to negotiate new commitments to reduce greenhouse gas emissions and combat climate change. The summit comes as scientists warn of increasingly severe weather patterns linked to global warming.",
    
    # Fake News
    "Breaking News: The moon is made of cheese, claims new scientific report.",
    "World Health Organization confirms that drinking bleach can cure COVID-19, urges people to start treatment immediately.",
    "Scientists discover a new continent hidden beneath the Pacific Ocean, larger than Australia.",
    "All global currencies will become obsolete next week due to the launch of a universal cryptocurrency backed by all major governments.",
    "The sun is expected to explode within the next 24 hours, according to a secret NASA report leaked by an anonymous source."
]

for i, text in enumerate(test_texts):
    result = FakeLense(text, bert_lense, bert_tokenizer, gpt_lense, gpt_tokenizer)
    print(f"News {i} : {result}\n")