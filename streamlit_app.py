import streamlit as st
from scripts.data_processing import initialize_data, process_texts, extract_topics
from scripts.model_initialization import initialize_models
from scripts.utils import retrieve_documents, vectorize_topics
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

def init_session_state():
    if 'initialized' not in st.session_state:
        st.session_state['initialized'] = True
        st.session_state['previous_prompt'] = ""
        st.session_state['previous_response'] = ""
        st.session_state['embed_model'], st.session_state['llm'], st.session_state['service_context'] = initialize_models()

        data_path = './data/'
        data = initialize_data(data_path)

        st.session_state['processed_data'] = process_texts(data)
        st.session_state['topics_data'] = extract_topics(st.session_state['processed_data'])
        st.session_state['vectorized_topics_data'] = vectorize_topics(st.session_state['topics_data'], st.session_state['embed_model'].encode)

if 'initialized' not in st.session_state:
    print("Initializing...")
    init_session_state()
print("Initialized")

st.markdown("## Ask your Questions!")

prompt = st.text_input('Enter your question:')

if prompt and prompt != st.session_state['previous_prompt']:
    st.session_state['previous_prompt'] = prompt
    best_match = retrieve_documents(prompt, st.session_state['vectorized_topics_data'], st.session_state['embed_model'].encode)
    print("Best match found")

    if best_match:
        csv_file = f"./data/{best_match[0]}_{best_match[1]}.csv"
        documents = SimpleDirectoryReader(input_files=[csv_file]).load_data()
        print("Documents read ",csv_file)
        index = VectorStoreIndex.from_documents(documents, service_context=st.session_state['service_context'])
        print("index")
        query_engine = index.as_query_engine()
        print("query_engine")
        response = query_engine.query(prompt)
        print("response")
        st.session_state['previous_response'] = response

st.markdown("## Response")
st.markdown(st.session_state['previous_response'])
