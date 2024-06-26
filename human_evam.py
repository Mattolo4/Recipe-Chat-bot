import pickle

def create_numbered_steps(lst):
    return '\n'.join([f"{i+1}. {step}" for i, step in enumerate(lst)])
with open("recipe_evaluation_all_fixed.pkl", 'rb') as f:
    recipes_evaluation_data = pickle.load(f)
print(recipes_evaluation_data.keys())
grade = {"cs":[], "lv":[], "pt": [], "ft":[]}
for i in range(75, len(recipes_evaluation_data["inputs"])):
    print("************\n")
    print(f"ROUND {i}:\n")
    print("Ingredients:\n")
    [print(ingr) for ingr in recipes_evaluation_data["inputs"][i]]

    print("\n")
    print("Cosine similarity recipe:")
    print(create_numbered_steps(list(recipes_evaluation_data["cosine_results"][i]["steps"])[0]))
    grade["cs"].append(float(input("Grade : ")))
    print("\n")
    print("Ingredients:\n")
    [print(ingr) for ingr in recipes_evaluation_data["inputs"][i]]
    print("Levenstein distance recipe:\n")
    print(create_numbered_steps(list(recipes_evaluation_data["levenstein_results"][i]["steps"])[0]))
    grade["lv"].append(float(input("Grade : ")))
    print("\n")
    print("Ingredients:\n")
    [print(ingr) for ingr in recipes_evaluation_data["inputs"][i]]
    print("Pretrained Llama 3 recipe:\n")
    pt_recipe = recipes_evaluation_data["pretrained_results"][i]
    if "### Response:" in pt_recipe:
        print(pt_recipe[pt_recipe.find("### Response:")+len("### Response: "):])
    else:
        print(pt_recipe)
    grade["pt"].append(float(input("Grade : ")))
    print("\n")
    print("Ingredients:\n")
    [print(ingr) for ingr in recipes_evaluation_data["inputs"][i]]
    print("Finetuned Llama 3 recipe:\n")
    ft_recipe = recipes_evaluation_data["finetuned_results"][i]
    if "### Response:" in ft_recipe:
        print(ft_recipe[ft_recipe.find("### Response:")+len("### Response: "):])
    else:
        print(ft_recipe)
    grade["ft"].append(float(input("Grade : ")))
    print("\n---------------\n")
    with open("human_grade.pkl", 'wb') as f:
        pickle.dump(grade, f)
    