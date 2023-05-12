# langchain-practices

Langchain practices with LLM, FAISS, and others

## Usage
Install dependencies
```
pip install -r requirements
```
Create `.env` in root folder and set API keys to environment variables
```
OPENAI_API_KEY=
HUGGINGFACEHUB_API_TOKEN=
SERPAPI_API_KEY=
```

In `main.py`, import methods from other python files or disable comment to run examples
```python
if __name__ == '__main__':
    load_api_key_from_env()

    msg = run_llm_davinci(message='Who are you?')
    print('davinci: ' + msg)
```

