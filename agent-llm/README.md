# run service  
uvicorn llm_service:app --host 0.0.0.0 --port 18888 --reload  
# run ui  
streamlit run ui.py
# ollama
Make sure Ollama is running: ollama serve