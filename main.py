import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from peft import PeftModel, PeftConfig

# Choose an embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the model configuration and weights with error handling
try:
    # Ensure the paths are correct and the files exist at the specified locations
    tokenizer = AutoTokenizer.from_pretrained("scaleszw/scales_ai")
    config = PeftConfig.from_pretrained("scaleszw/scales_ai")
    base_model = AutoModelForCausalLM.from_pretrained("unsloth/llama-3-8b-bnb-4bit")
    model = PeftModel.from_pretrained(base_model, "scaleszw/scales_ai")
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
    
    # Create a history container on the sidebar
    history_container = st.sidebar.container()
    
    # Add a title to the history container
    history_container.write("Conversation History")
    
    # Add a clickable list of previous conversations to the history container
    for i, message in enumerate(conversation_state.get_history()):
        history_container.write(f"{i+1} {message['content']}")
    
    # Add a horizontal rule to separate the history container from the rest of the sidebar
    st.sidebar.write("---")

# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    conversation_state.history = []

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA3 response
def generate_llama3_response(prompt):
    try:
        # Tokenize the prompt using your fine-tuned tokenizer
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids'][0].tolist()
        prompt_str = tokenizer.decode(input_ids, skip_special_tokens=True)
        
        # Retrieve context from the session state history
        context = ""
        if st.session_state.messages:
            context = " ".join([msg["content"] for msg in st.session_state.messages if msg["role"] == "user"])
        
        # Generate response
        outputs = model.generate(inputs['input_ids'], max_length=256)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return ""

# User-provided prompt
if prompt := st.text_input("You:", key="user_input"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate a new response if the last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.spinner("Thinking..."):
            response = generate_llama3_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Update the history container to include clickable links that scroll to the corresponding message
for i, message in enumerate(st.session_state.messages):
    history_container.write(f"{i+1} {message['content']}")
