import threading

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer
from transformers.utils.quantization_config import BitsAndBytesConfig

MODEL_NAME = "sbintuitions/sarashina2.2-3b-instruct-v0.1"


# NOTE: @st.cache_resource is essential here to prevent model reloading on every user interaction.
# Since Streamlit reruns the entire script each time.
@st.cache_resource
def load_model(model_name):
    """Load the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
    )
    return model, tokenizer


def _tokenize_prompts(tokenizer, prompts, device):
    """Tokenize the chat prompts."""

    formatted_text = tokenizer.apply_chat_template(
        prompts,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(
        formatted_text,
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    ).to(device)

    return model_inputs


def _create_generation_thread(model, streamer, inputs, tokenizer) -> threading.Thread:
    """Create a thread for model generation."""
    thread = threading.Thread(
        target=model.generate,
        kwargs=dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            streamer=streamer,
            max_new_tokens=512,
            use_cache=True,
            # Use randomness when choosing words
            do_sample=True,  # If False, the model always picks the most likely next word. Default False
            temperature=1.0,  # Higher temperature = more randomness. Default 1.0
            top_k=50,  # Limits the selection of next words. Default 50
            top_p=1.0,  # top_p=1.0 means no limit; top_p=0.9 would restrict to the top words that make up 90% of the probability. Default 1.0
        ),
    )
    return thread


def _extract_response(text: str) -> str:
    """Extract response from AI output."""
    text = text.split("<|assistant|>")[-1]
    text = text.split("</s>")[0]
    return text


def ai_response_stream(model, tokenizer, prompts):
    """Generator function for streaming AI responses."""
    model_inputs = _tokenize_prompts(tokenizer, prompts, model.device)

    # Prepare streamer
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=False
    )

    # Run model generation in a separate thread
    thread = _create_generation_thread(model, streamer, model_inputs, tokenizer)
    thread.start()

    # Streaming display
    for new_text in streamer:
        yield _extract_response(new_text)

    thread.join()


# Load the model and tokenizer
model, tokenizer = load_model(MODEL_NAME)


# Streamlit UI
st.title("Streamlit Local Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input and message handling
if prompt := st.chat_input("Please say something"):
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display AI response
    with st.chat_message("assistant"):
        response = st.write_stream(ai_response_stream(model, tokenizer, st.session_state.messages))
    st.session_state.messages.append({"role": "assistant", "content": response})
