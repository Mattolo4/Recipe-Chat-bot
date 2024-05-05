# Recipe-Chat-bot
Chat bot that help user to know what to cook depending on the ingredients and related conversations
# Ideas of things to implement : 
- [ ] Extract ingredients from the text of recipes (Could be trained on the dataset already available as a supervised model). For this we could use rules such as white list of the ingredients already in the dataset. We could also used a NER (Named Entity recognition trained on ingredients). Or use a Hidden Markov Model. Or try to use some embeddings of the words and then in the latent space find the cluster of ingredients and do some matching. Explore information retrieval techniques (Probabilistic Retrieval Models, LTR...) & search techniques in search engines or databases
- [ ] Generate a dataset using a LLM of examples of user prompt ("I have this, this and this and I wanna do this...") with the corresponding set of tags matching (ingredients, types, time to cook)
- [ ] Try to implement a LLM that generate a new recipe out of the user prompt
- [ ] Try to implement a LLM that generate a query in a databases of already existing recipes
- [ ] Implementing some feedback loop if we still have a lot of time
- [ ] Machine learning classifier to see if recipe good or bad given input
