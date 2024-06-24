import pandas as pd
import numpy as np
from collections import Counter
import re
import ast

def load_data(path="RAW_recipes.csv", preprocess_steps=False):
    #df_recipes = pd.read_csv(path, converters={'tags': lambda x: x[1:-1].split(','),
    #                                           'ingredients': lambda x: x[1:-1].split(',')})  #https://stackoverflow.com/questions/45758646/pandas-convert-string-into-list-of-strings
    #df_recipes = df_recipes.iloc[:100,:]
    #df_recipes["tags"] = df_recipes["tags"].apply(lambda lst: [re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', s).lower() for s in lst])   #chatgpt for regex...
    #df_recipes["ingredients"] = df_recipes["ingredients"].apply(lambda lst: [re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', s).lower() for s in lst])   #chatgpt for regex...
    
    df_recipes = pd.read_csv(path)
    df_recipes['tags'] = df_recipes['tags'].apply(ast.literal_eval)
    df_recipes['ingredients'] = df_recipes['ingredients'].apply(ast.literal_eval)

    if preprocess_steps:
        #    df_recipes["steps"] = df_recipes["steps"].apply(lambda lst: [re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', s).lower() for s in lst])   #chatgpt for regex...
        df_recipes['steps'] = df_recipes['steps'].apply(ast.literal_eval)

    return df_recipes

def filter_tags(df, tags):
    df_filtered = df.copy()
    tags_set = set(tags)
    df_filtered = df_filtered[~df_filtered['tags'].apply(lambda x: bool(tags_set.intersection(set(x))))]
    
    return df_filtered



if __name__=="__main__":
    df_all = load_data("RAW_recipes.csv")
    df_food = filter_tags(df_all,["beverages"])