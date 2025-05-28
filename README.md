# Streamlit and Gradio Comparison

## What is this?

This project demonstrates and compares the implementation of identical chat application features using both Streamlit and Gradio.

Both implementations create a simple LLM chat interface that runs a local Japanese LLM "Sarashina2.2-3b-instruct" for text generation.

## Requirements

- NVIDIA GPU with 8GB VRAM or higher
- CUDA
- uv

## Install

```sh
uv sync --frozen
```

## Run

### Streamlit

```sh
uv run streamlit run run_streamlit.py --server.fileWatcherType none
```

- `--server.fileWatcherType none` is used to disable the file watcher, which cause a RuntimeError.
    - See: https://github.com/streamlit/streamlit/issues/10992#issuecomment-2838270482

### Gradio

```sh
uv run run_gradio.py
```
