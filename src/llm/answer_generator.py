"""
LLM Answer Generation with Quality Scoring

Generates grounded answers using retrieved context with GPT-4o-mini.
Implements simplified 2-component quality scoring system.

Features:
- Strict grounding to source material (no hallucinations)
- Page number citations for traceability
- Simplified quality scoring (2 components):
  * Retrieval Quality (50%): How well chunks match the query
  * Completeness (50%): Has content + page citations

Quality classification: HIGH (≥0.7), MEDIUM (0.5-0.7), LOW (<0.5)

Uses openai_config module for API configuration.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

from openai_config.models import get_gpt_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerGenerator:
    """
    Generate grounded answers using LLM with quality scoring

    Uses GPT-4o-mini to generate answers strictly grounded in retrieved context.
    Implements simplified 2-component quality scoring for answer evaluation:

    1. Retrieval Quality (50%): How well retrieved chunks match the query
    2. Completeness (50%): Has substantive content + page citations

    Quality levels:
    - HIGH (≥0.7): Excellent answer with strong retrieval and complete content
    - MEDIUM (0.5-0.7): Good answer, may have minor gaps in retrieval or citations
    - LOW (<0.5): Weak answer, poor retrieval or incomplete content
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ):
        """
        Initialize answer generator

        Args:
            model: Model name (optional, uses config default)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
        """
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get LLM using model getter
        self.llm = get_gpt_model(model_name=model, temperature=temperature)
        self.model = model or "configured-model"

        logger.info(f"Initialized AnswerGenerator with model: {self.model}")

    def generate_answer(
        self, question: str, context: str, retrieved_chunks: List[Dict]
    ) -> Dict:
        """
        Generate answer for a question using retrieved context

        Args:
            question: User question
            context: Retrieved context string
            retrieved_chunks: List of retrieved chunk dicts for metadata

        Returns:
            Dict with answer, supporting_chunks, and metadata
        """
        logger.info(f"Generating answer for: {question[:100]}...")

        # Construct prompt
        prompt = self._construct_prompt(question, context)

        try:
            # Call LLM using LangChain
            messages = [
                ("system", self._get_system_prompt()),
                ("user", prompt),
            ]

            response = self.llm.invoke(messages)
            answer = response.content.strip()

            # Format supporting chunks with answer quality scores
            supporting_chunks = [
                {
                    "chunk_id": chunk["chunk_id"],
                    "page": chunk["page"],
                    "similarity_score": round(chunk.get("similarity_score", 0.0), 3)
                }
                for chunk in retrieved_chunks
            ]

            # Calculate hybrid answer quality score
            quality_metrics = self._calculate_hybrid_quality(
                question, answer, context, retrieved_chunks
            )

            avg_quality = quality_metrics["overall_score"]

            # Determine answer quality level
            if avg_quality >= 0.7:
                quality_level = "HIGH"
            elif avg_quality >= 0.5:
                quality_level = "MEDIUM"
            else:
                quality_level = "LOW"

            result = {
                "question": question,
                "answer": answer,
                "supporting_chunks": supporting_chunks,
                "answer_quality_score": round(avg_quality, 3),
                "answer_quality_level": quality_level,
                "quality_breakdown": quality_metrics,
                "metadata": {
                    "model": self.model,
                    "temperature": self.temperature,
                    "num_chunks_used": len(supporting_chunks),
                    "timestamp": datetime.now().isoformat(),
                },
            }

            logger.info(f"Generated answer ({len(answer)} chars)")
            return result

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "supporting_chunks": [],
                "metadata": {"error": str(e)},
            }

    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM"""
        return """You are an expert assistant specializing in public health education.

Your task is to answer questions based ONLY on the provided context from "Principles of Public Health" (1916 vintage textbook).

Guidelines:
1. Answer ONLY based on the information in the provided context
2. Be accurate, clear, and concise
3. If the context doesn't contain enough information, say so explicitly
4. Do NOT make up or infer information not present in the context
5. Do NOT use external knowledge or make assumptions beyond what's stated
6. Cite page numbers when mentioning specific information (e.g., "Page 25")
7. Use specific facts, examples, and principles from the context
8. Structure your answer logically with key points

IMPORTANT - Handling Historical Context:
- This is a 1916 textbook, so terminology and medical understanding reflect that era
- Focus on the principles and concepts described in the book
- Some medical terms may be outdated (e.g., "consumption" for tuberculosis)
- Statistical data represents early 1900s information

IMPORTANT - Handling Tables and Figures:
- The context may contain data from tables where structure was lost during OCR
- Numbers may appear without clear labels
- Look for patterns: disease names followed by death counts
- Figures are referenced as "Fig. X" or "Figure X"
- Table data often shows disease statistics, death rates, or year-over-year comparisons
- Use surrounding text and figure captions to interpret data
- Some tables have been manually corrected for clarity (e.g., structured death statistics)

CRITICAL - When data appears unstructured:
- Disease names may be separated from their corresponding numbers
- Look for context clues (e.g., "Deaths in 1907 from four diseases")
- Match items in the same order (first disease → first number, etc.)
- Use figure/table captions to understand what the data represents

Remember: Accuracy and faithfulness to the source material are paramount. Never hallucinate facts or figures. Always cite page numbers for specific claims."""

    def _construct_prompt(self, question: str, context: str) -> str:
        """
        Construct user prompt with question and context

        Args:
            question: User question
            context: Retrieved context

        Returns:
            Formatted prompt string
        """
        prompt = f"""Based on the following context from "Principles of Public Health" (1916), please answer the question.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Answer using ONLY the information provided in the context above
- Include specific statistics, percentages, or data points when available
- Cite page numbers for key information
- If the context is insufficient, clearly state what information is missing

ANSWER:"""

        return prompt

    def _calculate_hybrid_quality(
        self,
        question: str,
        answer: str,
        context: str,
        retrieved_chunks: List[Dict]
    ) -> Dict:
        """
        Calculate answer quality score based on retrieval and completeness

        Simplified 2-component scoring:
        1. Retrieval Quality (50%): How well chunks match the query
        2. Completeness (50%): Answer has content and cites sources

        Args:
            question: The user's question
            answer: The generated answer
            context: The context used for generation
            retrieved_chunks: List of retrieved chunks with similarity scores

        Returns:
            Dict with overall_score and component scores
        """
        import re

        # 1. Retrieval Quality (50% weight)
        # Average similarity of retrieved chunks
        retrieval_score = (
            sum(chunk.get("similarity_score", 0) for chunk in retrieved_chunks) /
            len(retrieved_chunks)
            if retrieved_chunks else 0.0
        )

        # 2. Answer Completeness (50% weight)
        # Check if answer has actual content and cites sources
        answer_lower = answer.lower()

        # Has substantive content (not just "no information available")
        has_content = (
            len(answer.split()) > 20 and
            "no information" not in answer_lower and
            "not found" not in answer_lower and
            "insufficient" not in answer_lower
        )

        # Cites page numbers from the source
        has_citations = bool(re.search(r'\(Page \d+\)', answer, re.IGNORECASE))

        # Completeness score: both criteria must be met for high score
        completeness_score = (0.6 if has_content else 0.2) + (0.4 if has_citations else 0.0)

        # Calculate weighted overall score (50/50 split)
        overall_score = 0.5 * retrieval_score + 0.5 * completeness_score

        # Return detailed breakdown
        return {
            "overall_score": round(overall_score, 3),
            "retrieval_score": round(retrieval_score, 3),
            "completeness_score": round(completeness_score, 3),
            "components": {
                "has_content": has_content,
                "has_citations": has_citations,
                "word_count": len(answer.split())
            }
        }

    def generate_batch_answers(
        self, questions: List[str], rag_pipeline
    ) -> List[Dict]:
        """
        Generate answers for multiple questions

        Args:
            questions: List of questions
            rag_pipeline: RAGPipeline instance for retrieval

        Returns:
            List of answer dicts
        """
        results = []

        for i, question in enumerate(questions, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing question {i}/{len(questions)}")
            logger.info(f"{'='*60}\n")

            # Retrieve context
            retrieved_chunks = rag_pipeline.retrieve(question)

            if not retrieved_chunks:
                logger.warning(f"No chunks retrieved for question {i}")
                results.append(
                    {
                        "question": question,
                        "answer": "No relevant information found in the document.",
                        "supporting_chunks": [],
                        "metadata": {},
                    }
                )
                continue

            # Construct context
            context = rag_pipeline.construct_context(retrieved_chunks)

            # Generate answer
            answer_result = self.generate_answer(question, context, retrieved_chunks)
            results.append(answer_result)

        return results


if __name__ == "__main__":
    # Example usage
    from src.rag.pipeline import RAGPipeline

    # Initialize
    rag_pipeline = RAGPipeline()
    generator = AnswerGenerator()

    # Test question
    question = "What are the main ways to fight disease germs according to the book?"

    # Retrieve and generate
    chunks = rag_pipeline.retrieve(question)
    context = rag_pipeline.construct_context(chunks)
    result = generator.generate_answer(question, context, chunks)

    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSupporting Chunks: {result['supporting_chunks']}")
