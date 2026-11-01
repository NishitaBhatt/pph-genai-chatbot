"""
Helper utility functions
"""

import json
import os
import logging
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


def save_results_json(results: List[Dict], output_path: str):
    """
    Save results to JSON file

    Args:
        results: List of answer result dicts
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved results to: {output_path}")


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to config file

    Returns:
        Config dict
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def log_query(
    question: str, answer: str, chunks: List[Dict], log_file: str = "output/query_logs.csv"
):
    """
    Log query and answer for traceability

    Args:
        question: User question
        answer: Generated answer
        chunks: Retrieved chunks
        log_file: Path to log file
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "num_chunks": len(chunks),
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "pages": [c["page"] for c in chunks],
    }

    # Append to CSV
    df = pd.DataFrame([log_entry])

    if os.path.exists(log_file):
        df.to_csv(log_file, mode="a", header=False, index=False)
    else:
        df.to_csv(log_file, mode="w", header=True, index=False)


def print_results(results: List[Dict]):
    """
    Pretty print results

    Args:
        results: List of answer result dicts
    """
    for i, result in enumerate(results, 1):
        print(f"\n{'='*80}")
        print(f"QUESTION {i}")
        print(f"{'='*80}")
        print(f"\n{result['question']}\n")
        print(f"{'-'*80}")
        print(f"ANSWER")
        print(f"{'-'*80}")
        print(f"\n{result['answer']}\n")
        print(f"{'-'*80}")
        print(f"SUPPORTING CHUNKS")
        print(f"{'-'*80}")
        for chunk in result.get("supporting_chunks", []):
            print(f"  - Chunk ID: {chunk['chunk_id']}, Page: {chunk['page']}")
        print()


def create_output_dir(output_dir: str = "output"):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
