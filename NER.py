import spacy
import random
from spacy.training.example import Example
import pandas as pd

def extract_ingredients_NER(user_input, model_path="ner_model"):
    nlp = spacy.load(model_path)
    results = []
    for sentence in user_input:
        doc = nlp(sentence)
        result = list(doc.ents)
        results.append([str(s).lower() for s in result])
    return results
    # print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])

if __name__=="__main__":
    print(extract_ingredients_NER("I have tomatoes and paprika. What do I cook? I also have some leftover pasta and a basil plant."))