# %%
import pandas as pd
import numpy as np
from collections import Counter
import re
from preprocessing import *
from white_list import *


# %%
def ingredient_to_int(ingredients_dict, ingredients):
    result = [ingredients_dict.setdefault(i, -1) for i in ingredients]
    result.sort()
    return result

def encode_ingredients_df(df,ingredients_dict):
    df['ingredients_encoded'] = df['ingredients'].apply(lambda x: ingredient_to_int(ingredients_dict, x))
    return df

def encode_ingredients_arr(arr,ingredients_dict):
    arr_encoded = ingredient_to_int(ingredients_dict, arr)
    return arr_encoded

""" def levenstein_dist(arr1,arr2,score):

    if not len(arr1) or not len(arr2):
        # one of them is empty
        return score
    
    if arr1[0] == arr2[0]:
        return levenstein_dist(arr1[1:], arr2[1:], score)
    
    else:
        return min(levenstein_dist(arr1[1:],    arr2,       score+1),
                   levenstein_dist(arr1,        arr2[1:],   score+1),
                   levenstein_dist(arr1[1:],    arr2[1:],   score+1)) """

def levenstein_dist(arr1,arr2,score):

    if not len(arr1) and not len(arr2):
        # both are empty
        return score
    
    if not len(arr1):
        # one of them is empty
        return score+len(arr2)
    
    if not len(arr2):
        # one of them is empty
        return score+len(arr1)
    
    if arr1[0] == arr2[0]:
        return levenstein_dist(arr1[1:], arr2[1:], score)
    
    while True:
        if arr1[0] < arr2[0]:
            return levenstein_dist(arr1[1:], arr2, score+1)
        else:
            return levenstein_dist(arr1, arr2[1:], score+1)
    
def get_recipe_levenstein_dists(df,ingredients):
    df_vecs = df['ingredients_encoded']
    dists = df_vecs.apply(lambda x: levenstein_dist(x, ingredients, 0))
    return dists

def get_recipes_levenstein(df,user_ingredients,ingredients_dict,n):
    
    if not "ingredients_encoded" in df.columns:
        df = encode_ingredients_df(df,ingredients_dict)

    user_ingredients_encoded = encode_ingredients_arr(user_ingredients,ingredients_dict)

    dists = get_recipe_levenstein_dists(df,user_ingredients_encoded)

    top_recipes_ids = dists.sort_values()[:n].index
    #top_recipes_ids = df.iloc[top_recipes.index]["ingredients"].index
    return df.iloc[top_recipes_ids]


if __name__ == "__main__":
    df_all = load_data("RAW_recipes.csv")
    df_food = filter_tags(df_all,["beverages"])
    df_food.reset_index(drop=True, inplace=True)

    ingredients_white_list = get_white_list(df_food)
    ingredients_dict = {string: index for index, string in enumerate(ingredients_white_list)}    #chatgpt

    user_ingredients = np.array(['garlic', 'onion', 'onions', 'tomatoes'])
    df_food = encode_ingredients_df(df_food,ingredients_dict)

    top_recipes = get_recipes_levenstein(df_food, user_ingredients, ingredients_dict, 10)
    
    print(top_recipes)


        