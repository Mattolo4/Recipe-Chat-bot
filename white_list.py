# %%
import pandas as pd
import numpy as np
from collections import Counter
import re
from preprocessing import *

MIN_INGREDIENT_APPEARANCES = 10

def preprocess_arr(arr_of_strings):
    chars_to_remove = ['!', '"', '%', '&', "'", '(', ')', '*', '-', '.', '/', '?']

    for char in chars_to_remove:
        arr_of_strings = np.char.replace(arr_of_strings, char, '') # chatgpt helped

    pattern = r'\d' # chatgpt provided (I can't do even simple regex from head I am sorry)

    arr_of_strings = np.array([re.sub(pattern, '', s) for s in arr_of_strings]) # get rid of numbers
    arr_of_strings = np.char.strip(arr_of_strings)  # get rid of leading and trailing white space
    return arr_of_strings

def get_word_sequences(text):
    return re.findall(r'\b[a-zA-Z]+\b', text)   # chatgpt

def white_list_user_input(white_list,user_input):

    if isinstance(user_input, str):
        mask = np.array([user_input.find(s) != -1 for s in white_list]) #chatpgt
        return white_list[mask]
    
    else:
        results = []
        for sentence in user_input:
            mask = np.array([sentence.find(s) != -1 for s in white_list]) #chatpgt
            results.append(white_list[mask])
        return results

def get_white_list(df):
    ingredients = df["ingredients"]#.iloc[0:1000]
    all_ingredients = np.concatenate(ingredients)

    ingredients_white_list,counts = np.unique(all_ingredients,return_counts=True)
    ingredients_white_list = ingredients_white_list[counts > MIN_INGREDIENT_APPEARANCES]
    ingredients_white_list = preprocess_arr(ingredients_white_list)

    return ingredients_white_list

if __name__ == "__main__":
    df_all = load_data("RAW_recipes.csv")
    df_food = filter_tags(df_all,["beverages"])
    df_food.reset_index(drop=True, inplace=True)

    ingredients_white_list = get_white_list(df_food)

    user_input = ["I have tomatoes, onions, and garlic. What can I cook with them?",
                "What dishes can I make with pasta, spinach, and cheese?",
                "I found some shrimp, bell peppers, and mushrooms in the fridge. Any recipe suggestions?",
                "My pantry has rice, beans, and tomatoes. What's a simple yet delicious meal I can prepare?",
                "I have eggs, bacon, and bread. How can I turn these into a tasty breakfast?",
                "What can I whip up with ground beef, bell peppers, and onions?",
                "I bought salmon, asparagus, and lemon. What's a good recipe for a healthy dinner?",
                "I've got tofu, broccoli, and soy sauce. Any suggestions for a vegetarian stir-fry?",
                "What desserts can I make with flour, sugar, and chocolate?",
                "I'm craving something sweet. What can I make with apples, cinnamon, and oats?"]
    user_input = preprocess_arr(user_input)

    print(white_list_user_input(ingredients_white_list, user_input))
