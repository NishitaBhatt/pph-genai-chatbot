# Deployment Files – Conversion Instructions

This folder contains all necessary files for deploying the PPH GenAI Chatbot. Files have been saved as `.txt` for easy transfer.

## Folder Structure to Create

```
pph_genai/
├── app/
│   └── streamlit_app.py          (from streamlit_app.txt)
├── src/
│   ├── __init__.py               (create empty file)
│   ├── rag/
│   │   ├── __init__.py           (create empty file)
│   │   ├── pipeline.py           (from rag_pipeline.txt)
│   │   └── hybrid_retrieval.py   (from hybrid_retrieval.txt)
│   └── llm/
│       ├── __init__.py           (create empty file)
│       └── answer_generator.py   (from answer_generator.txt)
├── utils/
│   ├── __init__.py               (create empty file)
│   └── helpers.py                (from helpers.txt)
├── openai_config/
│   ├── __init__.py               (create empty file)
│   └── utils.py                  (from openai_config_utils.txt)
├── vector_store/                 (copy entire folder or rebuild)
├── .streamlit/                   (if exists, copy folder)
├── requirements.txt
└── config.yaml
```

## Conversion Steps

1. **Create the folder structure above.**

2. **Rename `.txt` files back to `.py`:**
   - `streamlit_app.txt` → `app/streamlit_app.py`
   - `rag_pipeline.txt` → `src/rag/pipeline.py`
   - `hybrid_retrieval.txt` → `src/rag/hybrid_retrieval.py`
   - `answer_generator.txt` → `src/llm/answer_generator.py`
   - `helpers.txt` → `utils/helpers.py`
   - `openai_config_utils.txt` → `openai_config/utils.py`

3. **Keep as-is:**
   - `requirements.txt`
   - `config.yaml`
   - `.streamlit/` folder (if exists)

4. **Create empty `__init__.py` files in:**
   - `src/`
   - `src/rag/`
   - `src/llm/`
   - `utils/`
   - `openai_config/`

5. **Copy the `vector_store/` folder** (if you have pre-built embeddings).

6. **Create a `.env` file or use Streamlit secrets for:**
   - `OPENAI_API_KEY`
   - Any other environment variables

## For Streamlit Cloud Deployment

1. Push to GitHub with the proper structure.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub repo.
4. Select `app/streamlit_app.py` as the main file.
5. Add secrets in the Streamlit dashboard (Settings → Secrets):

   ```toml
   OPENAI_API_KEY = "your-key-here"
   OPENAI_ENDPOINT = "your-endpoint"
   OPENAI_API_VERSION = "your-version"
   ```

6. Deploy!

## Notes

- Chunker was excluded as requested (not rebuilding).
- OCR components excluded (not needed for chatbot).
- Main pipeline script excluded (not needed for deployment).
- Vector store should be included if you want to avoid rebuilding embeddings.
- Make sure all `__init__.py` files are created for proper Python imports.
