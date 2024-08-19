from fake_lense import * 

# Detection Phase
bert_lense, bert_tokenizer = load_model_and_tokenizer('./model/bert_lense', AutoModelForSequenceClassification)
gpt_lense, gpt_tokenizer = load_model_and_tokenizer('./model/gpt_lense', AutoModelForCausalLM)

test_cases = [
    # Truth News
   "In the wake of the recent election, residents of Amherst gathered at the local common for a peaceful vigil, expressing solidarity and resolve. The event, which took place at Edwards Church, drew a large crowd from across the community. Speakers addressed the need for unity and moving forward with strength. The atmosphere was one of reflection and hope, as people discussed the implications of the election results and what steps can be taken next.",
   "Long before Hillary Clinton, Victoria Woodhull was the first woman to run for president, setting a precedent nearly 150 years ago. Woodhull, a progressive activist, advocated for women's suffrage, civil rights, and free love. Her candidacy was groundbreaking, challenging the societal norms of the time. Today, Woodhull's legacy lives on as women continue to break barriers in politics and beyond, inspired by her pioneering efforts.",
   "The community was left in shock after the tragic death of FBI Special Agent David Raynor, who was found dead alongside his family in what authorities believe to be a murder-suicide. Raynor had been involved in several high-profile investigations, and his sudden death has raised many questions. Colleagues remember him as a dedicated officer who served with distinction. The investigation into the circumstances surrounding the incident continues.",

    # Fake News
   "Contrary to initial reports, new evidence suggests that Michael Brown was not the innocent victim portrayed by the media. Witnesses now reveal that Brown had attempted to flee the scene after robbing a store and was shot while struggling with Officer Darren Wilson. Despite these revelations, mainstream media outlets continue to push a narrative that fuels public outrage and division, ignoring the complexities of the case.",
   "In a shocking twist, FBI Special Agent David Raynor, who was reportedly investigating a connection between Hillary Clinton and a satanic pedophile ring, was found dead in his home. While official reports suggest a murder-suicide, conspiracy theorists claim that Raynor was silenced to protect powerful figures involved in the ring. The Clinton campaign has denied these allegations, dismissing them as baseless conspiracy theories.",
   "A former government insider has come forward with explosive claims that a secret plan is in place to control the population through implanted microchips. According to the whistleblower, these microchips will be introduced under the guise of health and security measures, but their true purpose is to monitor and manipulate citizens. The source alleges that this plan has been in development for years and involves coordination between governments and tech companies.",
]

for i, text in enumerate(test_cases):
    result = FakeLense(text, bert_lense, bert_tokenizer, gpt_lense, gpt_tokenizer)
    print(f"News {i} : {result}\n")
