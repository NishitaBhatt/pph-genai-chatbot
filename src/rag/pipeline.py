"""
RAG Pipeline Implementation

Handles vector storage, semantic retrieval, and context construction.
Uses ChromaDB for vector storage with OpenAI embeddings.

Note: This provides base semantic search. For better retrieval, use
HybridRetriever (src/rag/hybrid_retrieval.py) which combines this
semantic search with keyword matching (BM25).

Uses openai_config module for API configuration.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings

from openai_config.models import get_embedding_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG pipeline for vector storage and semantic retrieval

    Provides:
    - ChromaDB vector store with persistence
    - Semantic similarity search (cosine distance)
    - Context construction from retrieved chunks

    For production use, wrap with HybridRetriever for better results
    (combines semantic + keyword BM25 search).
    """

    def __init__(
        self,
        persist_directory: str = "vector_store",
        collection_name: str = "health_genai_chunks",
        embedding_model: Optional[any] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ):
        """
        Initialize RAG pipeline

        Args:
            persist_directory: Directory to persist vector store
            collection_name: Name of the collection
            embedding_model: Embedding model (if None, uses ChromaDB default)
            top_k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity score
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Initialize or get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Created new collection: {collection_name}")

        self.embedding_model = embedding_model

    def index_chunks(self, chunks: List[Dict], batch_size: int = 100):
        """
        Index chunks into vector store

        Args:
            chunks: List of chunk dicts with text and metadata
            batch_size: Batch size for indexing
        """
        logger.info(f"Indexing {len(chunks)} chunks...")

        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []

        for chunk in chunks:
            documents.append(chunk["text"])
            metadatas.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "page": chunk["page"],
                    "token_count": chunk.get("token_count", 0),
                    "char_count": chunk.get("char_count", 0),
                }
            )
            ids.append(chunk["chunk_id"])

        # Add to collection in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            self.collection.add(
                documents=batch_docs, metadatas=batch_meta, ids=batch_ids
            )

            if (i + batch_size) % 500 == 0:
                logger.info(f"Indexed {min(i + batch_size, len(documents))} chunks")

        logger.info(f"Successfully indexed all {len(chunks)} chunks")

    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a query

        Args:
            query: User query
            top_k: Number of chunks to retrieve (overrides default)

        Returns:
            List of retrieved chunk dicts with text, metadata, and scores
        """
        k = top_k or self.top_k

        logger.info(f"Retrieving top-{k} chunks for query: {query[:100]}...")

        # Query the collection
        results = self.collection.query(query_texts=[query], n_results=k)

        # Format results
        retrieved_chunks = []

        if results and results["documents"] and len(results["documents"][0]) > 0:
            for i, (doc, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                # Convert distance to similarity score (cosine similarity)
                similarity = 1 - distance

                if similarity < self.similarity_threshold:
                    continue

                retrieved_chunks.append(
                    {
                        "rank": i + 1,
                        "chunk_id": metadata["chunk_id"],
                        "text": doc,
                        "page": metadata["page"],
                        "similarity_score": round(similarity, 4),
                        "distance": round(distance, 4),
                    }
                )

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks above threshold")
        return retrieved_chunks

    def construct_context(
        self, retrieved_chunks: List[Dict], max_tokens: int = 3000
    ) -> str:
        """
        Construct context string from retrieved chunks

        Args:
            retrieved_chunks: List of retrieved chunk dicts
            max_tokens: Maximum tokens for context

        Returns:
            Formatted context string
        """
        if not retrieved_chunks:
            return "No relevant context found."

        context_parts = []
        total_chars = 0
        approx_max_chars = max_tokens * 4  # Rough approximation

        for chunk in retrieved_chunks:
            chunk_text = (
                f"[Page {chunk['page']}, Chunk {chunk['chunk_id']}]\n"
                f"{chunk['text']}\n"
            )

            if total_chars + len(chunk_text) > approx_max_chars:
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

        context = "\n---\n\n".join(context_parts)
        return context

    def clear_collection(self):
        """Clear all data from the collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")

    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "persist_directory": self.persist_directory,
        }


if __name__ == "__main__":
    # Example usage
    import json

    # Load chunks
    with open("output/chunks.json") as f:
        chunks = json.load(f)

    # Initialize pipeline
    pipeline = RAGPipeline()

    # Index chunks
    pipeline.index_chunks(chunks)

    # Test retrieval
    query = "How can we fight disease germs?"
    results = pipeline.retrieve(query)

    print(f"\nQuery: {query}")
    print(f"\nRetrieved {len(results)} chunks:\n")
    for r in results:
        print(f"Rank {r['rank']}: Page {r['page']} - Score: {r['similarity_score']}")
        print(f"Text: {r['text'][:200]}...\n")
