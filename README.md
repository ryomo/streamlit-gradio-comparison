# Streamlit and Gradio Comparison

## What is this?

This project demonstrates and compares the implementation of identical application features using both Streamlit and Gradio.

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
