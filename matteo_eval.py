# %%
import pandas as pd
import numpy as np
from collections import Counter
import re


# %% [markdown]
# ## Ingredients extraction
# #### LLAMA3 set up
# 1. Download from https://ollama.com/
# 2. Install
# 3. Click here http://localhost:11434/ to check if it’s running (it should appear 'Ollama is running')
# 4. Run in a terminal: `ollama run llama3`
# 
# If u dont have: `pip install langchain-community`

from langchain_community.llms import Ollama
from collections import Counter
from preprocessing import *
import re
from tqdm import tqdm

def extractIngredients(sentences):
    llm = Ollama(model="llama3")
    all_ingredients = []

    for sentence in tqdm(sentences):
        output = ''
        prompt = f"Output the ingredients from the following sentence without any comments, separated only by commas: {sentence}"
        for chunks in llm.stream(prompt):
            output += chunks

        #pattern = r'\b[a-zA-Z]+\b'  # Simple pattern to match words
        #ingredients = re.findall(pattern, output.lower())
        all_ingredients.append(output)
    
    return all_ingredients

ground_truth = [["winter squash"],["eggs","milk"],["onions", "ground beef"]]
user_inputs = [
    "What can I cook with winter squash, mexican seasoning, mixed spice, honey, butter, olive oil, and salt?",
    "Can you suggest a recipe using prepared pizza crust, sausage patty, eggs, milk, salt and pepper, and cheese?",
    "What dish can I make with ground beef, yellow onions, diced tomatoes, tomato paste, tomato soup, rotel tomatoes, kidney beans, water, chili powder, ground cumin, salt, lettuce, and cheddar cheese?",
    "I'd like to cook something with spreadable cheese with garlic and herbs, new potatoes, shallots, parsley, tarragon, olive oil, red wine vinegar, salt, pepper, red bell pepper, and yellow bell pepper. Any ideas?",
    "How can I use tomato juice, apple cider vinegar, sugar, salt, pepper, clove oil, cinnamon oil, and dry mustard in a recipe?",
    "Do you have a recipe that includes milk, vanilla ice cream, frozen apple juice concentrate, and apple?",
    "What can I make with fennel seeds, green olives, ripe olives, garlic, peppercorn, orange rind, orange juice, red chile, and extra virgin olive oil?",
    "Can you suggest a dish using pork spareribs, soy sauce, fresh garlic, fresh ginger, chili powder, fresh coarse ground black pepper, salt, fresh cilantro leaves, tomato sauce, brown sugar, yellow onion, white vinegar, honey, A.1. original sauce, liquid smoke, cracked black pepper, cumin, dry mustard, cinnamon sticks, orange juice, mirin, and water?",
    "I have chocolate sandwich style cookies, chocolate syrup, vanilla ice cream, bananas, strawberry ice cream, and whipped cream. What can I make?",
    "What can I bake with sugar, unsalted butter, bananas, eggs, fresh lemon juice, orange rind, cake flour, baking soda, and salt?",
    "Can you suggest a recipe using whole berry cranberry sauce, sour cream, and prepared horseradish?",
    "I need a recipe that includes vanilla wafers, butter, powdered sugar, eggs, whipping cream, strawberry, and walnuts.",
    "How can I use great northern bean, chicken bouillon cubes, dark brown sugar, molasses, cornstarch, onion, garlic powder, mustard powder, chili powder, salt, black pepper, bacon, and water in a recipe?",
    "What can I cook with collard greens, brown sugar, molasses, hot sauce, whiskey, and ham hock?",
    "Do you have a recipe that includes gentian root, scullcap herb, burnet root, wood bethony, and spearmint?",
    "What can I make with lean pork chops, flour, salt, dry mustard, garlic powder, oil, and chicken rice soup?",
    "Can you suggest a dish using egg roll wrap, whole green chilies, cheese, cornstarch, and oil?",
    "I'd like to cook something with butterscotch chips, Chinese noodles, and salted peanuts. Any ideas?",
    "How can I use celery, onion, ground pork, soy sauce, beef broth, cooking oil, and hamburger buns in a recipe?",
    "What can I make with canola oil, onion, garlic, cauliflower, potatoes, vegetable bouillon cubes, water, salt free herb and spice seasoning mix, ground coriander, great northern bean, salt and pepper, broccoli floret, escarole, green peas, red bell pepper, and fresh herb?",
    "Can you suggest a recipe using water, salt, boiling potatoes, fresh spinach leaves, unsalted butter, coarse salt, fresh ground black pepper, and nutmeg?",
    "I need a dish that includes onion, scallion, apple juice, olive oil, spinach, fresh parsley, celery, broth, rolled oats, salt, dried thyme, and white pepper.",
    "What can I cook with boneless skinless chicken breast halves, condensed cream of chicken soup, egg, seasoning salt, all-purpose flour, cornstarch, garlic powder, paprika, salt and pepper, and oil?",
    "Can you suggest a recipe using all-purpose flour, granulated sugar, baking powder, salt, vanilla extract, egg, milk, vegetable oil, bread, brown sugar, ground cinnamon, butter, and powdered sugar?",
    "How can I use butter, lemon juice, salt, white pepper, and egg yolks in a dish?",
    "What can I make with ground black pepper, ground ginger, ground coriander, ground cumin, ground turmeric, and black cumin?",
    "I'd like to cook something with vegetarian ground beef, garlic, onion, jalapenos, green pepper, celery, kidney beans, diced tomatoes, chili powder, black pepper, salt, and red pepper flakes. Any ideas?",
    "What can I make with beef stew meat, water, tomatoes, beef bouillon cube, onion, dried parsley, salt, ground thyme, ground pepper, zucchini, cabbage, garbanzo beans, elbow macaroni, and parmesan cheese?",
    "Do you have a recipe that includes red potatoes and margarine, and rosemary?",
    "What dish can I make with unsalted butter, carrot, onion, celery, broccoli stem, dried thyme, dried oregano, dried sweet basil leaves, dry white wine, chicken stock, Worcestershire sauce, Tabasco sauce, smoked chicken, black beans, broccoli floret, heavy cream, salt & fresh ground pepper, and cornstarch?",
    "What can I bake with butter, sugar, vanilla, eggs, all-purpose flour, baking cocoa, baking powder, salt, and miniature peppermint patties?",
    "Can you suggest a recipe using ground beef, onion, tomato sauce, taco sauce, salt, pepper, Tabasco sauce, hot chili pepper, cornmeal, whole kernel corn, sliced ripe olives, and cheddar cheese?",
    "How can I use butter, dry ranch dressing mix, and French bread in a recipe?",
    "I need a dish that includes ground venison, egg substitute, non-fat powdered milk, water, fresh breadcrumb, onion, salt, black pepper, dry mustard, and Worcestershire sauce.",
    "What can I make with milk, frozen juice concentrate, and plain yogurt?",
    "Can you suggest a recipe using low sodium chicken broth, diced tomatoes, zucchini, corn, potatoes, wax beans, green beans, and carrots?",
    "What dish can I make with frozen chopped spinach, egg, salt, black pepper, onion, sharp cheddar cheese, condensed cream of mushroom soup, and crouton?",
    "What can I cook with red potatoes, green onion, diced pimentos, fat-free mayonnaise, plain low-fat yogurt, low-fat sour cream, sugar, prepared mustard, white wine vinegar, salt, pepper, celery seed, and garlic?",
    "Can you suggest a recipe using frozen chopped spinach, eggs, garlic powder, soft breadcrumbs, oregano, margarine, sage, and onion?",
    "What dish can I make with ground beef, onion, frozen vegetables, cream of mushroom soup, condensed cream of mushroom & garlic soup, salt & pepper, cooking oil, and cornbread mix?",
    "I'd like to cook something with angel hair pasta, toasted sesame oil, soy sauce, honey, garlic, green onions, toasted sesame seeds, and stir fry vegetables. Any ideas?",
    "Can you suggest a recipe using all-purpose flour, buckwheat flour, unsweetened cocoa, baking powder, baking soda, salt, ground cinnamon, ground cloves, honey, sugar, eggs, yam, low-fat buttermilk, orange rind, orange juice, canola oil, brown sugar, flour, cinnamon, butter, and pecans?",
    "What can I bake with all-purpose flour, buckwheat flour, unsweetened cocoa, baking powder, baking soda, salt, ground cinnamon, ground cloves, sorghum, eggs, yam, low-fat buttermilk, orange rind, orange juice, canola oil, raisins, boiling water, and granulated sugar?",
    "How can I use ground beef, sugar, prepared yellow mustard, beer, cayenne, garlic, salt & pepper, and American cheese in a recipe?",
    "What can I make with tri-color spiral pasta, dill pickles, ripe olives, green onion, chives, sweet pepper, water chestnut, tomatoes, ham, cheese, olive oil, cider vinegar, onion powder, garlic powder, salt and pepper, and Italian seasoning?",
    "Do you have a recipe that includes frozen French fries, oil, salt & freshly ground black pepper, parmesan cheese, oregano, basil, and flat leaf parsley?",
    "What dish can I make with sandwich bun, Good Seasonings Italian salad dressing mix, butter, deli turkey, ham, pepperoni, cheddar cheese, Swiss cheese, and mozzarella cheese?",
    "How can I use shortening, icing sugar, vanilla, all-purpose flour, baking powder, sugar, eggs, salt, milk, and butter in a recipe?",
    "What can I bake with yellow cake mix, vanilla instant pudding mix, nutmeg, cinnamon, eggs, oil, water, crushed pineapple, carrot, pecans, and coconut?",
    "Can you suggest a recipe using whole kernel corn, onion, red bell pepper, butter, Jiffy corn muffin mix, egg, heavy cream, jalapenos, sharp cheddar cheese, and honey?",
    "What dish can I make with flour, water, dry yeast, milk, sugar, eggs, vegetable oil, baking soda, baking powder, salt, cinnamon, vanilla, crushed pineapple, raisins, nuts, butter, and brown sugar?",
    "I'd like to cook something with Italian sausage, ground beef, garlic, dried basil, salt, whole tomato, tomato paste, ricotta cheese, cottage cheese, parmesan cheese, parsley flakes, eggs, ground black pepper, lasagna noodles, mozzarella cheese, and mushrooms. Any ideas?",
    "Can you suggest a recipe using margarine, sugar, eggs, lemon extract, self-rising flour, evaporated milk, and flaked coconut?",
    "How can I use chicken tenders, egg, butter, flour, milk, dried parsley, lemon, salt, pepper, and heavy cream in a dish?",
    "What can I make with green onions, snow peas, mung bean sprout, carrot, water chestnut, oil, soy sauce, cornstarch, pork butt, flour, garlic powder, and ginger?",
    "What dish can I make with hamburger bun, pork patty, egg, and cheese slice?",
    "Can you suggest a recipe using instant tea, sugar, cinnamon, cloves, and nutmeg?",
    "What can I make with ground beef, cream of mushroom soup, evaporated milk, beef broth, garlic powder, onion powder, parsley, pepper, and noodles?",
    "How can I use chicken stock, potatoes, corn kernels, flour, heavy cream, black pepper, ground thyme, onion, and butter in a dish?",
    "What dish can I make with cream of mushroom soup, cream of celery soup, frozen peas, celery, milk, thyme, paprika, salt, black pepper, and canned salmon?",
    "Can you suggest a recipe using chicken, garlic, soy sauce, dried oregano, salt, fresh ground black pepper, canola oil, tomato, onion, chicken broth, cumin, fresh cilantro, avocado, lime, tortilla, and shredded cheddar cheese?",
    "I'd like to cook something with red potatoes, carrots, red onion, zucchini, butter, fresh parsley, and salt. Any ideas?",
    "What can I bake with graham cracker crust, sweetened condensed milk, lime juice, sour cream, and lime peel?",
    "Can you suggest a recipe using frozen apple juice concentrate, chicken broth, Dijon mustard, molasses, and cornstarch?",
    "What dish can I make with vanilla instant pudding mix, vanilla extract, milk, cream cheese, powdered sugar, lemon, and condensed milk?",
    "How can I use potatoes, onions, and cheddar cheese in a recipe?",
    "What can I cook with broccoli florets, butter, cream of chicken soup, American cheese, cheddar cheese, milk, salt, and pepper?",
    "Can you suggest a recipe using ground beef, onion, potato, chili powder, salt, tomato, and lettuce?",
    "What dish can I make with eggs, powdered sugar, butter, evaporated milk, and vanilla?",
    "How can I use graham cracker crust, sweetened condensed milk, lime juice, sour cream, and lime peel in a dish?",
    "What can I make with chicken broth, fresh mushrooms, butter, flour, cream, salt, fresh ground black pepper, dried dill, and lemon juice?",
    "Can you suggest a recipe using vanilla instant pudding mix, vanilla extract, milk, cream cheese, powdered sugar, lemon, and condensed milk?",
    "How can I use pork patty, hamburger bun, egg, and cheese slice in a dish?"
]


