import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def vectorize_topics(topics_data, embed_func):
    vectorized_topics_data = {}

    for main_cat, subcats in topics_data.items():
        vectorized_topics_data[main_cat] = {}
        for subcat, words in subcats.items():
            combined_text = " ".join(words)
            embed = embed_func(combined_text)
            vectorized_topics_data[main_cat][subcat] = normalize_vector(embed)

    return vectorized_topics_data

def retrieve_documents(prompt, vectorized_topics_data, embed_func):
    prompt_vector = normalize_vector(embed_func(prompt))
    best_match = None
    best_score = -1

    for category, subcategories in vectorized_topics_data.items():
        for subcategory, vector in subcategories.items():
            similarity = cosine_similarity([prompt_vector], [vector])
            if similarity > best_score:
                best_score = similarity
                best_match = (category, subcategory)

    return best_match
