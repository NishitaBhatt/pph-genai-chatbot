"""
Streamlit Chatbot Application for PPH GenAI Q&A

Interactive web interface for querying "Principles of Public Health" book.

Features:
- Hybrid search (semantic + keyword BM25)
- Multi-component answer quality scoring
- Adjustable retrieval parameters (top_k, similarity threshold, hybrid balance)
- Visual quality indicators (HIGH/MEDIUM/LOW)
- Supporting evidence with page citations
- JSON export functionality

Usage:
    streamlit run app/streamlit_app.py
    # Access at http://localhost:8501
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import json
from datetime import datetime

from src.rag.pipeline import RAGPipeline
from src.rag.hybrid_retrieval import HybridRetriever
from src.llm.answer_generator import AnswerGenerator
from utils.helpers import load_config, log_query

# Configure OpenAI
from openai_config.utils import config_env, set_token

# Page configuration
st.set_page_config(
    page_title="PPH GenAI Q&A Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .question-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .chunk-box {
        background-color: #f9f9f9;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_components():
    """
    Initialize RAG pipeline, hybrid retriever, and answer generator (cached)

    Returns:
        tuple: (rag_pipeline, hybrid_retriever, answer_generator, config)
            - rag_pipeline: Base RAG pipeline with ChromaDB
            - hybrid_retriever: Hybrid search (semantic + BM25)
            - answer_generator: LLM answer generator with quality scoring
            - config: Configuration dict from config.yaml
    """
    try:
        # Configure OpenAI API
        config_env()
        set_token()

        config = load_config("config.yaml")

        rag_pipeline = RAGPipeline(
            persist_directory=config["vector_store"]["persist_directory"],
            collection_name=config["vector_store"]["collection_name"],
            top_k=config["rag"]["retrieval"]["top_k"],
            similarity_threshold=config["rag"]["retrieval"]["similarity_threshold"]
        )

        # Load chunks for hybrid search
        chunks_path = os.path.join(config["output"]["dir"], "chunks.json")
        with open(chunks_path) as f:
            chunks = json.load(f)

        # Initialize hybrid retriever (combines semantic + keyword search)
        hybrid_retriever = HybridRetriever(
            rag_pipeline=rag_pipeline,
            all_chunks=chunks,
            alpha=0.5  # Default: equal weight for semantic and keyword
        )

        answer_generator = AnswerGenerator(
            model=config["llm"]["model"],
            temperature=config["llm"]["temperature"],
            max_tokens=config["llm"]["max_tokens"]
        )

        return rag_pipeline, hybrid_retriever, answer_generator, config

    except Exception as e:
        st.error(f"Error initializing components: {e}")
        st.info("Make sure you've run `python main.py` first to set up the vector store.")
        st.stop()


def main():
    """
    Main Streamlit application

    Provides interactive Q&A interface with:
    - Question input (custom or sample questions)
    - Hybrid search retrieval
    - Quality-scored answers with breakdown
    - Supporting evidence display
    - Adjustable parameters (top_k, threshold, alpha)
    - JSON export
    """

    # Header
    st.markdown('<p class="main-header">📚 Principles of Public Health Q&A Assistant</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Initialize components
    with st.spinner("Loading RAG pipeline with hybrid search..."):
        rag_pipeline, hybrid_retriever, answer_generator, config = initialize_components()

    # Sidebar
    with st.sidebar:
        st.header("📚 About")
        st.write("""
        This chatbot answers questions about **Principles of Public Health**
        using a RAG (Retrieval-Augmented Generation) pipeline.

        **Features:**
        - OCR-extracted text from the book
        - **Hybrid search** (semantic + keyword matching)
        - Semantic chunking for better context
        - Grounded answers with supporting evidence
        - Answer quality scoring
        - Full traceability to source pages
        """)

        st.header("⚙️ Settings")

        top_k = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=10,
            value=config["rag"]["retrieval"]["top_k"]
        )

        similarity_threshold = st.slider(
            "Similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=config["rag"]["retrieval"]["similarity_threshold"],
            step=0.05
        )

        alpha = st.slider(
            "Hybrid search balance",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0.0 = keyword only, 1.0 = semantic only, 0.5 = balanced"
        )

        # Update pipeline settings
        rag_pipeline.top_k = top_k
        rag_pipeline.similarity_threshold = similarity_threshold
        hybrid_retriever.set_alpha(alpha)

        st.header("📊 Collection Stats")
        stats = rag_pipeline.get_collection_stats()
        st.metric("Total Chunks", stats["total_chunks"])

        st.header("💡 Sample Questions")
        sample_questions = config["questions"]

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Ask a Question")

        # Question input methods
        input_method = st.radio(
            "Input method:",
            ["Type custom question", "Select sample question"],
            horizontal=True
        )

        if input_method == "Type custom question":
            question = st.text_area(
                "Enter your question:",
                height=100,
                placeholder="e.g., What are the main ways to fight disease germs?"
            )
        else:
            question = st.selectbox(
                "Select a sample question:",
                sample_questions
            )

        # Generate answer button
        if st.button("🔍 Generate Answer", type="primary", use_container_width=True):
            if not question:
                st.warning("Please enter a question first.")
            else:
                with st.spinner("Retrieving relevant information and generating answer..."):
                    # Retrieve chunks using hybrid search (semantic + keyword)
                    retrieved_chunks = hybrid_retriever.retrieve(
                        question,
                        top_k=top_k,
                        min_score=0.0  # Apply threshold in the answer generator
                    )

                    if not retrieved_chunks:
                        st.error("No relevant information found. Try adjusting the similarity threshold or hybrid balance.")
                    else:
                        # Construct context
                        context = rag_pipeline.construct_context(retrieved_chunks)

                        # Generate answer
                        result = answer_generator.generate_answer(
                            question,
                            context,
                            retrieved_chunks
                        )

                        # Store in session state
                        st.session_state["last_result"] = result

                        # Log query
                        if config["output"]["log_queries"]:
                            log_query(
                                question,
                                result["answer"],
                                retrieved_chunks,
                                config["output"]["log_file"]
                            )

    with col2:
        st.subheader("Export Options")

        if "last_result" in st.session_state:
            result_json = json.dumps(
                st.session_state["last_result"],
                indent=2,
                ensure_ascii=False
            )

            st.download_button(
                label="📥 Download as JSON",
                data=result_json,
                file_name=f"answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

    # Display results
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]

        st.markdown("---")
        st.subheader("📝 Results")

        # Question
        st.markdown('<div class="question-box">', unsafe_allow_html=True)
        st.markdown("**Question:**")
        st.write(result["question"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Answer
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.markdown("**Answer:**")
        st.write(result["answer"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Answer Quality Score
        if "answer_quality_score" in result:
            quality_score = result["answer_quality_score"]
            quality_level = result.get("answer_quality_level", "UNKNOWN")

            # Color based on quality level
            color = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(quality_level, "⚪")

            st.markdown(f"**Answer Quality:** {color} {quality_level} (Score: {quality_score:.3f})")

            # Show quality breakdown if available
            if "quality_breakdown" in result:
                breakdown = result["quality_breakdown"]
                with st.expander("📊 Quality Score Breakdown"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Retrieval Quality", f"{breakdown['retrieval_score']:.3f}", help="How well chunks match the query (50% weight)")
                    with col2:
                        st.metric("Completeness", f"{breakdown['completeness_score']:.3f}", help="Has content + citations (50% weight)")

                    components = breakdown.get('components', {})
                    st.caption(f"✓ Has content: {components.get('has_content', False)} | ✓ Has citations: {components.get('has_citations', False)} | Word count: {components.get('word_count', 0)}")

        # Supporting chunks
        st.markdown("**📄 Supporting Evidence:**")

        with st.expander(f"View {len(result['supporting_chunks'])} supporting chunks"):
            for i, chunk in enumerate(result["supporting_chunks"], 1):
                st.markdown(f"**Chunk {i}: {chunk['chunk_id']}** (Page {chunk['page']})")

                # Find full chunk text
                chunk_details = next(
                    (c for c in retrieved_chunks if c["chunk_id"] == chunk["chunk_id"]),
                    None
                )

                if chunk_details:
                    st.markdown('<div class="chunk-box">', unsafe_allow_html=True)
                    st.text(chunk_details["text"][:500] + ("..." if len(chunk_details["text"]) > 500 else ""))

                    # Show hybrid search scores
                    similarity = chunk.get('similarity_score', chunk_details.get('similarity_score', 0))
                    semantic = chunk_details.get('semantic_score', 0)
                    bm25 = chunk_details.get('bm25_score', 0)

                    if semantic > 0 or bm25 > 0:
                        st.caption(f"📊 Hybrid Score: {similarity:.3f} (Semantic: {semantic:.3f}, Keyword: {bm25:.3f})")
                    else:
                        st.caption(f"📊 Similarity Score: {similarity:.3f}")

                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("---")

        # Metadata
        with st.expander("🔍 View Metadata"):
            st.json(result.get("metadata", {}))

    # Footer
    st.markdown("---")
    st.caption("Built with Streamlit • Powered by Azure OpenAI • RAG Pipeline with Grounded Answers")


if __name__ == "__main__":
    main()
