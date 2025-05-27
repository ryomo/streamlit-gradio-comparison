import threading
import gradio as gr
from functools import lru_cache
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer
from transformers.utils.quantization_config import BitsAndBytesConfig

MODEL_NAME = "sbintuitions/sarashina2.2-3b-instruct-v0.1"

# NOTE: @lru_cache is currently inefective since load_model() is only called once.
@lru_cache(maxsize=1)
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


model, tokenizer = load_model(MODEL_NAME)


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


def _create_generation_thread(model, streamer, inputs, tokenizer):
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
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
        ),
    )
    return thread


def _extract_response(text: str) -> str:
    """Extract response from AI output."""
    text = text.split("<|assistant|>")[-1]
    text = text.split("</s>")[0]
    return text


def chat_function(message, history):
    """Main chat function for Gradio."""

    # Convert Gradio history format to messages format
    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current message
    messages.append({"role": "user", "content": message})

    # Prepare model inputs
    model_inputs = _tokenize_prompts(tokenizer, messages, model.device)

    # Prepare streamer
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=False
    )

    # Run model generation in a separate thread
    thread = _create_generation_thread(model, streamer, model_inputs, tokenizer)
    thread.start()

    # Stream the response
    response = ""
    for new_text in streamer:
        extracted = _extract_response(new_text)
        if extracted:
            response += extracted
            yield response

    thread.join()


# Create Gradio interface
with gr.Blocks(title="Gradio Local Chat") as demo:
    gr.Markdown("# Gradio Local Chat")

    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        type="messages",
        height=600
    )

    with gr.Row():
        msg = gr.Textbox(
            scale=4,
            placeholder="Please say something",
            container=False,
            show_label=False
        )
        submit_btn = gr.Button("Send", scale=1)

    # Clear button
    clear_btn = gr.Button("Clear Chat History", variant="secondary")

    def user_input(message, history):
        """Handle user input."""
        return "", history + [{"role": "user", "content": message}]

    def bot_response(history):
        """Generate bot response."""
        user_message = history[-1]["content"]
        # Remove the last entry and pass history without it
        chat_history = history[:-1]

        # Stream the response
        for response in chat_function(user_message, chat_history):
            # Update the last message with assistant response
            if len(history) > 0 and history[-1]["role"] == "user":
                updated_history = history[:-1] + [
                    history[-1],
                    {"role": "assistant", "content": response}
                ]
            else:
                updated_history = history + [{"role": "assistant", "content": response}]
            yield updated_history

    def clear_chat():
        """Clear chat history."""
        return []

    # Event handlers
    msg.submit(user_input, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, [chatbot], [chatbot]
    )
    submit_btn.click(user_input, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, [chatbot], [chatbot]
    )
    clear_btn.click(clear_chat, outputs=[chatbot])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
