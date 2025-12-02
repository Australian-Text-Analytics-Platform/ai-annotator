"""
Utility Functions for Streamlit App

Helper functions for CSV parsing, schema validation, and result formatting.
"""
import json
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
from io import StringIO


def parse_csv(uploaded_file) -> pd.DataFrame:
    """
    Parse an uploaded CSV file into a pandas DataFrame

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        pandas DataFrame

    Raises:
        Exception: If CSV parsing fails
    """
    try:
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        raise Exception(f"Failed to parse CSV file: {str(e)}")


def validate_schema(schema_dict: Dict[str, Any], technique: str) -> Tuple[bool, Optional[str]]:
    """
    Validate schema against technique requirements

    Args:
        schema_dict: Schema dictionary to validate
        technique: Classification technique ("zero_shot" or "few_shot")

    Returns:
        Tuple of (is_valid, error_message)
        error_message is None if schema is valid
    """
    try:
        # Check for 'classes' field
        if 'classes' not in schema_dict:
            return False, "Schema must contain 'classes' field"

        if not isinstance(schema_dict['classes'], list):
            return False, "'classes' must be an array"

        if len(schema_dict['classes']) == 0:
            return False, "'classes' array cannot be empty"

        # Validate each class
        for idx, cls in enumerate(schema_dict['classes']):
            if not isinstance(cls, dict):
                return False, f"Class at index {idx} must be an object"

            if 'name' not in cls:
                return False, f"Class at index {idx} missing 'name' field"

            if 'description' not in cls:
                return False, f"Class at index {idx} missing 'description' field"

            if not isinstance(cls['name'], str) or not cls['name'].strip():
                return False, f"Class at index {idx} has invalid 'name' (must be non-empty string)"

            if not isinstance(cls['description'], str) or not cls['description'].strip():
                return False, f"Class at index {idx} has invalid 'description' (must be non-empty string)"

        # Additional validation for few-shot
        if technique == "few_shot":
            if 'examples' not in schema_dict:
                return False, "Few-shot schema must contain 'examples' field"

            if not isinstance(schema_dict['examples'], list):
                return False, "'examples' must be an array"

            if len(schema_dict['examples']) == 0:
                return False, "'examples' array cannot be empty for few-shot"

            # Collect valid class names
            class_names = {cls['name'] for cls in schema_dict['classes']}

            # Validate each example
            for idx, ex in enumerate(schema_dict['examples']):
                if not isinstance(ex, dict):
                    return False, f"Example at index {idx} must be an object"

                if 'query' not in ex:
                    return False, f"Example at index {idx} missing 'query' field"

                if 'classification' not in ex:
                    return False, f"Example at index {idx} missing 'classification' field"

                if not isinstance(ex['query'], str) or not ex['query'].strip():
                    return False, f"Example at index {idx} has invalid 'query' (must be non-empty string)"

                if not isinstance(ex['classification'], str) or not ex['classification'].strip():
                    return False, f"Example at index {idx} has invalid 'classification' (must be non-empty string)"

                # Check if classification matches a defined class
                if ex['classification'] not in class_names:
                    return False, f"Example at index {idx} has classification '{ex['classification']}' not in defined classes"

        return True, None

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_json(json_str: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Validate and parse JSON string

    Args:
        json_str: JSON string to validate

    Returns:
        Tuple of (is_valid, parsed_dict, error_message)
        parsed_dict is None if JSON is invalid
        error_message is None if JSON is valid
    """
    try:
        parsed = json.loads(json_str)
        if not isinstance(parsed, dict):
            return False, None, "JSON must be an object (not an array or primitive)"
        return True, parsed, None
    except json.JSONDecodeError as e:
        return False, None, f"Invalid JSON: {str(e)}"
    except Exception as e:
        return False, None, f"Error parsing JSON: {str(e)}"


def format_results_for_download(results: List[Dict[str, Any]]) -> str:
    """
    Format classification results as CSV string for download

    Args:
        results: List of classification result items

    Returns:
        CSV string
    """
    if not results:
        return "index,text,classification,confidence,reasoning,reasoning_content\n"

    # Create DataFrame from results with all fields
    data = []
    for r in results:
        row = {
            'index': r['index'],
            'text': r['text'],
            'classification': r['classification'],
            'confidence': r.get('confidence'),
            'reasoning': r.get('reasoning'),
            'reasoning_content': r.get('reasoning_content')
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Convert to CSV
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()


def should_warn_large_batch(num_texts: int, threshold: int = 100) -> bool:
    """
    Determine if a large batch warning should be shown

    Args:
        num_texts: Number of texts in the batch
        threshold: Threshold for large batch warning (default: 100)

    Returns:
        True if warning should be shown
    """
    return num_texts > threshold


def build_classification_request(
    texts: List[str],
    user_schema: Dict[str, Any],
    provider: str,
    model: str,
    technique: str,
    temperature: float,
    top_p: float,
    llm_api_key: Optional[str] = None,
    llm_endpoint: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    enable_reasoning: bool = False,
    max_reasoning_chars: int = 150
) -> Dict[str, Any]:
    """
    Build request payload for /classify/batch endpoint

    Args:
        texts: List of texts to classify
        user_schema: Classification schema
        provider: LLM provider ("openai", "gemini", "anthropic", or "ollama")
        model: Model name
        technique: Classification technique
        temperature: Temperature value
        top_p: Top P value
        llm_api_key: Optional LLM API key (for OpenAI, Gemini, Anthropic)
        llm_endpoint: Optional LLM endpoint (for Ollama)
        reasoning_effort: Optional reasoning mode ("low", "medium", "high")
        enable_reasoning: Enable reasoning output (default: False)
        max_reasoning_chars: Maximum characters for reasoning (default: 150)

    Returns:
        Request payload dictionary
    """
    payload = {
        "texts": texts,
        "user_schema": user_schema,
        "provider": provider,
        "model": model,
        "technique": technique,
        "modifier": "no_modifier",
        "temperature": temperature,
        "top_p": top_p,
        "reasoning_effort": reasoning_effort,
        "enable_reasoning": enable_reasoning,
        "max_reasoning_chars": max_reasoning_chars
    }

    # Add provider-specific fields
    if provider in ["openai", "gemini", "anthropic"] and llm_api_key:
        payload["llm_api_key"] = llm_api_key
    elif provider == "ollama" and llm_endpoint:
        payload["llm_endpoint"] = llm_endpoint

    return payload
