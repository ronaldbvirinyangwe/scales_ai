import os
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from peft import PeftModel, PeftConfig
from huggingface_hub import login
import requests

# Set the Hugging Face token
token = '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'
os.environ['HUGGINGFACE_HUB_TOKEN'] = token

# Log in to Hugging Face
login(token)

# Choose an embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the model configuration and weights with error handling
try:
    tokenizer = AutoTokenizer.from_pretrained("scaleszw/scales_ai", use_auth_token=True)
    config = PeftConfig.from_pretrained("scaleszw/scales_ai", use_auth_token=True)
    base_model = AutoModelForCausalLM.from_pretrained("unsloth/llama-3-8b-bnb-4bit", use_auth_token=True)
    model = PeftModel.from_pretrained(base_model, "scaleszw/scales_ai", use_auth_token=True)
except ValueError as e:
    st.error(f"Error loading model: {e}")
    st.stop()
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()

class ConversationState:
    def __init__(self):
        self.history = []
        self.entities = {}

    def add_message(self, message):
        self.history.append(message)

    def update_entities(self, entities):
        self.entities.update(entities)

    def get_history(self):
        return self.history

    def get_entities(self):
        return self.entities

conversation_state = ConversationState()

# Google Custom Search API setup
GOOGLE_API_KEY = "AIzaSyDgjfLk1DJFm-rZtA5mkNWjRk4vsIyO_iIa"
SEARCH_ENGINE_ID = "122c61a18e46b44f5"

def google_search(query):
    try:
        response = requests.get(
            'https://www.googleapis.com/customsearch/v1',
            params={'q': query, 'key': GOOGLE_API_KEY, 'cx': SEARCH_ENGINE_ID},
        )
        results = response.json()
        return results['items'] if 'items' in results else []
    except Exception as e:
        st.error(f"Error during Google search: {e}")
        return []

# App title
st.set_page_config(page_title="Scales AI", layout="centered", initial_sidebar_state="auto")
st.title("Welcome to Scales AI")

st.markdown("""
<style>
.st-emotion-cache-uf99v8.ea3mdgi8 { background: #0E1117; }
.st-emotion-cache-1kyxreq.e115fcil2 { position: relative; top: 0px; left: 10px; z-index: 1; }
.st-emotion-cache-rde19y.ezrtsby2, .st-emotion-cache-zq5wmm.ezrtsby0, .st-emotion-cache-6q9sum.ef3psqc4 { background: url('Scales Technologies Abstract Logo.svg'); }
img { top:0; width:200px; background-size: cover; max-width: 100%; height: auto; margin-right: 20px; cursor: pointer; }
.st-emotion-cache-1wbqy5l.e17vllj40, .st-emotion-cache-klqnuk.en6cib64, .st-emotion-cache-ch5dnh.ef3psqc5, .st-emotion-cache-1u4fkce.en6cib62, .st-emotion-cache-czk5ss.e16jpq800, .st-emotion-cache-1dp5vir.ezrtsby1 { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("Scales Technologies Final Draft-01.svg")
    
    history_container = st.sidebar.container()
    history_container.write("Conversation History")
    
    for i, message in enumerate(conversation_state.get_history()):
        history_container.write(f"{i+1} {message['content']}")
    
    st.sidebar.write("---")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    conversation_state.history = []

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

def generate_llama3_response(prompt):
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(inputs['input_ids'], max_length=256)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return ""

if prompt := st.text_input("You:", key="user_input"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.spinner("Thinking..."):
            response = generate_llama3_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})

for i, message in enumerate(st.session_state.messages):
    history_container.write(f"{i+1} {message['content']}")
