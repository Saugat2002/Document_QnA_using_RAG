from transformers import BertTokenizer, BertModel
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
from bertopic import BERTopic
import nltk
from nltk.corpus import stopwords
import re
import torch
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from transformers import AutoModelForCausalLM, AutoConfig
from huggingface_hub import login
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import ServiceContext
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
import streamlit as st
from transformers import BitsAndBytesConfig

def get_sentence_embedding(text):
    return st.session_state.embed_model.encode(text)

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

def retrieve_documents(prompt, vectorized_topics_data):
    prompt_vector = normalize_vector(get_sentence_embedding(prompt))
    best_match = None
    best_score = -1

    for category, subcategories in vectorized_topics_data.items():
        for subcategory, vector in subcategories.items():
            similarity = cosine_similarity([prompt_vector], [vector])
            if similarity > best_score:
                best_score = similarity
                best_match = (category, subcategory)

    return best_match

# Initialization
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.previous_prompt = ""
    st.session_state.previous_response = ""  # Initialize previous_response

    nltk.download('stopwords')

    data_path = './data/'
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

    st.session_state.embed_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    st.session_state.bert_model = BERTopic(language="english", nr_topics=5, min_topic_size=2)
    st.session_state.stop_words = set(stopwords.words('english'))

    st.session_state.vectorized_data = {}

    for category, subcategories in data.items():
        st.session_state.vectorized_data[category] = {}
        for subcategory, texts in subcategories.items():
            st.session_state.vectorized_data[category][subcategory] = [get_sentence_embedding(text) for text in texts]

    processed_data = {
        'Finance': {},
        'Marketing': {},
        'HR': {}
    }

    for category, subcategories in data.items():
        for subcategory, docs in subcategories.items():
            docs_without_stopwords = []
            for doc in docs:
                words = doc.split()
                filtered_words = [word for word in words if word.lower() not in st.session_state.stop_words]
                filtered_doc = ' '.join(filtered_words)
                docs_without_stopwords.append(filtered_doc)
            processed_data[category][subcategory] = docs_without_stopwords

    topics_data = {
        'Finance': {},
        'Marketing': {},
        'HR': {}
    }

    topics_list = []

    for category, subcategories in processed_data.items():
        for sub_category, docs in subcategories.items():
            st.session_state.bert_model = BERTopic(language="english", nr_topics=5, min_topic_size=2)
            topics, probs = st.session_state.bert_model.fit_transform(docs)
            topic_freq = st.session_state.bert_model.get_topic_freq()
            topic_values = topic_freq['Topic'].tolist()
            for topic in topic_values:
                topics_tuples = st.session_state.bert_model.get_topic(topic)
                topics_list.append([topic for topic, weight in topics_tuples])
            topics_data[category][sub_category] = list(set(flatten(topics_list)))

    st.session_state.vectorized_topics_data = {}

    for main_cat, subcats in topics_data.items():
        if main_cat not in st.session_state.vectorized_topics_data:
            st.session_state.vectorized_topics_data[main_cat] = {}
        for subcat, words in subcats.items():
            combined_text = " ".join(words)
            embed = get_sentence_embedding(combined_text)
            st.session_state.vectorized_topics_data[main_cat][subcat] = normalize_vector(embed)

    system_prompt = """
    You are a Q&A assistant. Your goal is to answer questions as
    accurately as possible based on the instructions and context provided.
    """
    query_wrapper_prompt = SimpleInputPrompt("{query_str}")

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model_kwargs = {
        "torch_dtype": torch.float16,
        "quantization_config": quantization_config,
    }

    st.session_state.llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        device_map="cuda",
        model_kwargs=model_kwargs
    )

    st.session_state.embed_model_2 = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

    st.session_state.service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=st.session_state.llm,
        embed_model=st.session_state.embed_model_2
    )

    print("Setup Complete")
else:
    print("Setup Complete")


# prompt = st.text_input('Enter your prompt:', value=st.session_state.previous_prompt)
# if prompt and prompt != st.session_state.previous_prompt:
#     st.session_state.previous_prompt = prompt
#     best_match = retrieve_documents(prompt, st.session_state.vectorized_topics_data)
#     print(f"Best Match: {best_match}")
#     if best_match:
#         csv_file = f"./data/{best_match[0]}_{best_match[1]}.csv"
#         documents = SimpleDirectoryReader(input_files=[csv_file]).load_data()
#         index = VectorStoreIndex.from_documents(documents, service_context=st.session_state.service_context)
#         query_engine = index.as_query_engine()
#         response = query_engine.query(prompt)
#         st.session_state.previous_response = response


# st.markdown("## Response")
# st.markdown(st.session_state.previous_response)

if 'previous_prompt' not in st.session_state:
    st.session_state.previous_prompt = ""
    st.session_state.previous_response = ""

st.markdown("## Ask your Questions!")

# Input prompt from user
prompt = st.text_input('Enter your question:')

if prompt and prompt!= st.session_state.previous_prompt:
    st.session_state.previous_prompt = prompt
    best_match = retrieve_documents(prompt, st.session_state.vectorized_topics_data)
    print(f"Best Match: {best_match}")
    if best_match:
        csv_file = f"./data/{best_match[0]}_{best_match[1]}.csv"
        documents = SimpleDirectoryReader(input_files=[csv_file]).load_data()
        index = VectorStoreIndex.from_documents(documents, service_context=st.session_state.service_context)
        query_engine = index.as_query_engine()
        response = query_engine.query(prompt)
        st.session_state.previous_response = response

        # Display the response
st.markdown("## Response")
st.markdown(st.session_state.previous_response)

