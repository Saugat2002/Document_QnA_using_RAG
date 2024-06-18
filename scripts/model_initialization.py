import torch
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from llama_index.core import ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.legacy.embeddings.langchain import LangchainEmbedding

def initialize_models():
    embed_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
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

    llm = HuggingFaceLLM(
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

    embed_model_2 = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embed_model_2
    )

    return embed_model, llm, service_context
