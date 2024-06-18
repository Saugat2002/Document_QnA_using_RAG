import pandas as pd
import os
from collections import defaultdict
from nltk.corpus import stopwords
from bertopic import BERTopic
import nltk

nltk.download('stopwords')

def initialize_data(data_path):
    data = {
        'Finance': {
            'Financial_Planning': pd.read_csv(os.path.join(data_path, 'Finance_Financial_Planning.csv'))['content'].tolist(),
            'Financial_Reporting': pd.read_csv(os.path.join(data_path, 'Finance_Financial_Reporting.csv'))['content'].tolist(),
            'Risk_Management': pd.read_csv(os.path.join(data_path, 'Finance_Risk_Management.csv'))['content'].tolist()
        },
        'Marketing': {
            'Social_Media_Marketing': pd.read_csv(os.path.join(data_path, 'Marketing_Social_Media_Marketing.csv'))['content'].tolist(),
            'Customer_Engagement': pd.read_csv(os.path.join(data_path, 'Marketing_Customer_Engagement.csv'))['content'].tolist(),
            'Market_Research': pd.read_csv(os.path.join(data_path, 'Marketing_Market_Research.csv'))['content'].tolist()
        },
        'HR': {
            'Recruitment': pd.read_csv(os.path.join(data_path, 'HR_Recruitment.csv'))['content'].tolist(),
            'Training': pd.read_csv(os.path.join(data_path, 'HR_Training.csv'))['content'].tolist(),
            'Performance_Management': pd.read_csv(os.path.join(data_path, 'HR_Performance_Management.csv'))['content'].tolist()
        }
    }
    return data

def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

def process_texts(data):
    stop_words = set(stopwords.words('english'))
    processed_data = defaultdict(dict)
    
    for category, subcategories in data.items():
        for subcategory, texts in subcategories.items():
            filtered_texts = [' '.join([word for word in doc.split() if word.lower() not in stop_words]) for doc in texts]
            processed_data[category][subcategory] = filtered_texts
    
    return processed_data

def extract_topics(processed_data):
    topics_data = defaultdict(dict)
    topics_list = []
    
    for category, subcategories in processed_data.items():
        for subcategory, docs in subcategories.items():
            bert_model = BERTopic(language="english", nr_topics=5, min_topic_size=2)
            topics, probs = bert_model.fit_transform(docs)
            topic_freq = bert_model.get_topic_freq()
            topic_values = topic_freq['Topic'].tolist()
            
            for topic in topic_values:
                topics_tuples = bert_model.get_topic(topic)
                topics_list.append([topic for topic, weight in topics_tuples])
            
            topics_data[category][subcategory] = list(set(flatten(topics_list)))
    
    return topics_data
