task_to_benchmark = {
    # GLUE
    "cola": "glue",
    "mnli": "glue",
    "mrpc": "glue",
    "qnli": "glue",
    "qqp":  "glue",
    "sst2": "glue",
    "stsb": "glue",
    "wnli": "glue",
    # SuperGLUE excluding "rte": ("premise", "hypothesis"), since it is the same as in GLUE
    "boolq":   "super_glue",
    "cb":      "super_glue", 
    "wic":     "super_glue", 
    "copa":    "super_glue",
    "multirc": "super_glue",
    "wsc.fixed":     "super_glue",
    "rte":     "super_glue",
    "record":  "super_glue",
    # Summerization,
    "xsum": None,
    "samsum": None,
    "multi_news": None,
    # Generation
    "e2e_nlg_cleaned": None,
    "web_nlg_en": "gem",
    "common_gen": None,
    # Sentiment Classification
    "imdb": None,
    "yelp_review_full": None,
    # open-domain QA
    "social_i_qa": None,
    "wiki_qa": None,
    "fullwiki": "hotpot_qa",
    # Coreference Resolution
    "winogrande_debiased": "winogrande",
    "quoref": None,
    # Sentence completion
    "hellaswag": None,
    "story_cloze": "story_cloze",
    "anli_r1": "anli",
    "anli_r2": "anli",
    "anli_r3": "anli",
}

task_to_instruction_template = {
    
    # Sentence completion

    "hellaswag" : '''{{ ctx }}...

      How does the description likely end?


      A: {{ endings[0] }}


      B: {{ endings[1] }}


      C: {{ endings[2] }}


      D: {{ endings[3] }}

      ||| {{ answer_choices[label | int()] }}''',

    "story_cloze": '''
    {{input_sentence_1}} {{input_sentence_2}} {{input_sentence_3}} {{input_sentence_4}}
          What is a possible continuation for the story given the following options ? \n
          A: {{sentence_quiz1}} \n
          B: {{sentence_quiz2}}
          ||| {{answer_choices[answer_right_ending -1]}}
    ''', 

    # Conference Resolution
    "winogrande_debiased" : '''{{sentence}}

      What does the _ in the above sentence refer to? A: {{ option1 }} or B: {{ option2 }}? 
      ||| {{answer_choices[answer | int - 1]}}''', 

    "quoref" : '''The answer to the question: {{question}} is inside the article: {{context}},
      can you guess it ?


      |||

      {{answers.text | choice}}''',

    # NLI Tasks

    "anli_r1": '''Given that {{premise}} Does it follow that {{hypothesis}} Yes, no, or maybe?
      ||| {{ answer_choices[label] }}''',

    "anli_r2": '''Given that {{premise}} Does it follow that {{hypothesis}} Yes, no, or maybe?
      ||| {{ answer_choices[label] }}''',
    
    "anli_r3": ''''Given that {{premise}} Does it follow that {{hypothesis}} Yes, no, or maybe?
      ||| {{ answer_choices[label] }}''',

    # For GLUE benchmark
    "cola" : '''{{sentence}}

      Is this example grammatically correct and sensible?

      |||

      {{ answer_choices[label] }}''',
    
    "mnli" : '''{{premise}} Based on the previous passage, is it true that "{{hypothesis}}"?
      Yes, no, or maybe? ||| {{ answer_choices[label] }}''',

    "mrpc" : '''Does the sentence

      {{sentence1}}

      paraphrase (that is, mean the same thing as) this sentence?

      {{sentence2}}

      |||

      {{ answer_choices[label] }}''',

    "qnli" : '''Can you answer the question "{{question}}" based only on the following:

      {{sentence}}

      |||

      {{answer_choices[label]}}''',

    "qqp" : '''I received the questions "{{question1}}" and "{{question2}}". Are they
      duplicates? ||| {{ answer_choices[label] }}''',

    "sst2" : '''I''m reading a review that says "{{sentence}}".


      Do you think the review is {{"positive"}} or {{"negative"}}? ||| {{ answer_choices[label]
      }}''',

    "stsb" : '''Please rate how similar these two sentences are from {{"0.0"}} to {{"5.0"}}.

      Sentence A: {{sentence1}}

      Sentence B: {{sentence2}}

      |||

      {{ (((5*label) | round )/5) }}''',

    "wnli" : '''Assume that the following is true:

      {{sentence1}}

      Does this mean that "{{sentence2}}"?

      |||

      {{answer_choices[label]}}''',

    # For superGLUE 

    "boolq" : '''{{ passage }} \n\nHaving read that, I wonder {{ question }}? |||\n{% if
       label != -1 %}\n{{ answer_choices[label] }} \n{% endif %}''',

    "cb" : '''{{premise}} Based on the previous passage, is it true that "{{hypothesis}}"?
      Yes, no, or maybe? ||| {% if label !=-1 %}{{ answer_choices[label] }}{% endif
      %}''',

    "wic" : '''Does the word "{{word}}" have the same meaning in these two sentences?

      {{sentence1}}

      {{sentence2}}

      ||| {% if label != -1%}

      {{answer_choices[label]}}

      {% endif %}''',

    "copa" : '''Pick the more likely continuation to the following sentence:

      {{ premise }} {% if question == "cause" %} as a result of: {% else %} as a consequence:
      {% endif %}

      - {{choice1}}

      - {{choice2}} ||| {% if label != -1 %}{{ answer_choices[label] }}{%endif%}''',

    "multirc" : '''{{paragraph}}


      Question: {{question}}

      I found this answer "{{answer}}". Is that correct? Yes or no?

      |||

      {% if label != -1 %}{{answer_choices[label]}}{% endif %}''',

    "wsc.fixed" : '''{{ text }} In the previous sentence, does the pronoun "{{ span2_text.lower()
      }}" refer to {{ span1_text }}? Yes or no? ||| {% if label != -1 %}{{ answer_choices[label]
      }}{% endif %}''',

    "rte" : '''{{premise}} Using only the above description and what you know about the
      world, is "{{hypothesis}}" definitely correct? Yes or no? ||| {% if label !=
      -1 %}{{ answer_choices[label] }}{% endif %}''',

    "record" : '''Summary:\n\n- {{ passage.split(\"@highlight\")[1:] | join(\"\\n- \") }}
       \n\nArticle:\n\n{{ passage.split(\"@highlight\")[0] }}\n\nNow that you've
       read the article, please write a new sentence to add to it.\n\n||| {% if (
       answers | length ) > 0 %}{{ query | replace(\"@placeholder\", answers | choice)
       }} {% endif %}''',

    # Summerization
    "xsum" : '''{{document}}


      ===


      Write a summary of the text above : ||| {{summary}}''',

    "samsum" : '''Summarize this dialogue: {{dialogue}} |||

      {{summary}}''',

    "multi_news" : '''{% set docs = document.split("3ed2dface8203c4c9dfb1a5dc58e41e0||") | reject("equalto",
      "") | list %}

      What are the key points across these news articles:

      {% for doc in docs %}


      Article: {{doc}}

      {% endfor %}

      |||

      {{summary[2:]}}''',

    # Generation
    "e2e_nlg_cleaned" : '''Combine all of the following data into a concise and grammatically correct
      text:

      {% for feature in meaning_representation.split("]") %} {% set key = feature.split("[")[0].replace(",","")
      %} {% set value = feature.replace(",","").replace(key+"[", '''''''') %}

      {% if value != "" %} {{key}} : {{value}} {% endif %}

      {%- endfor %}

      ||| {{human_reference}}''',

    "web_nlg_en" : '''Verbalize the following triples separated by a comma: {{old_input | join(\", \")}} ||| {{target}}''',

    "common_gen" : '''Ignoring the order of the concepts: {{ concepts | join(\", \") }}; \n\
      Generate a sentence with all the concepts :\n|||\n{{target}}''',

    # Sentiment Classification
    "imdb" : '''{{text}} Did the reviewer find this movie {{"good or bad"}}? ||| {{ answer_choices
      [label] }}''',

    "yelp_review_full" : '''{{ text }}

      ===

      Based on that, my rating is ||| {{ answer_choices[label] }}''',

    # open-domain QA
    "social_i_qa" : '''I heard that {{context}}


      And I was wondering {{question}}


      |||


      {{answer_choices[label | int - 1]}}''',

    "wiki_qa" : '''I am verifying the answers generated by an automatic system to the following
      question: {{question}}

      Suggested answer: {{answer}}

      Should I validate this answer?

      |||

      {{answer_choices[label]}}''',

    "fullwiki" : '''Answer the following question, \"{{question}}\", using the information\
      \ provided below.\n\n{% for sents in context.sentences %}\n  - {{sents | join(\"\
      \")}}\n{% endfor %}\n||| \n{{answer}}''',


}
    
task_is_generative_task = {
    # Other tasks
    "hellaswag": False,
    "story_cloze": False,
    "anli_r1": False,
    "anli_r2": False,
    "anli_r3": False,

    # GLUE
    "cola": False,
    "mnli": False,
    "mrpc": False,
    "qnli": False,
    "qqp":  False,
    "sst2": False,
    "stsb": False,
    "wnli": False,
    # SuperGLUE excluding "rte": ("premise", "hypothesis"), since it is the same as in GLUE
    "boolq":    False,
    "cb":       False, 
    "wic":      False, 
    "copa":     False,
    "multirc":  False,
    "wsc.fixed":False,
    "rte":      False,
    "record":   False,
    # Tweet Eval
    'emoji':           False,
    'emotion':         False,
    'hate':            False,
    'irony':           False,
    'offensive':       False,
    'sentiment':       False,
    'stance_abortion': False,
    'stance_atheism':  False,
    'stance_climate':  False,
    'stance_feminist': False,
    'stance_hillary':  False,
    # NLI Tasks
    "anli_r1": False,
    "anli_r2": False,
    "anli_r3": False,
    # Summerization,
    "xsum": False,
    "samsum": False,
    "multi_news": False,
    # Generation
    "e2e_nlg_cleaned": True,
    "web_nlg_en": True,
    "common_gen": True,
    # Sentiment Classification
    "imdb": False,
    "yelp_review_full": False,
    # open-domain QA
    "social_i_qa": False,
    "wiki_qa": False,
    "fullwiki": False,
    # Coreference Resolution
    "winogrande_debiased": False,
    "quoref": False
}