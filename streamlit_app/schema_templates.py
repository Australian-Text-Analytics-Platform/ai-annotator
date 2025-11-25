"""
Schema Templates for Classification

Provides example schema templates for different classification tasks.
"""
from typing import Dict, List, Any


# Template definitions
TEMPLATES: Dict[str, Dict[str, Any]] = {
    "Sentiment Analysis (Zero-shot)": {
        "classes": [
            {
                "name": "Positive",
                "description": "Describes a positive sentiment or event"
            },
            {
                "name": "Negative",
                "description": "Describes a negative sentiment or event"
            },
            {
                "name": "Neutral",
                "description": "Neither positive nor negative, factual or objective"
            }
        ]
    },
    "Topic Classification (Zero-shot)": {
        "classes": [
            {
                "name": "Technology",
                "description": "Technology, software, hardware, computing"
            },
            {
                "name": "Health",
                "description": "Health, medicine, wellness, medical topics"
            },
            {
                "name": "Business",
                "description": "Business, finance, economics, corporate"
            },
            {
                "name": "Entertainment",
                "description": "Entertainment, movies, music, pop culture"
            },
            {
                "name": "Other",
                "description": "Topics not fitting other categories"
            }
        ]
    },
    "Sentiment with Examples (Few-shot)": {
        "classes": [
            {
                "name": "Positive",
                "description": "Describes a positive sentiment or event"
            },
            {
                "name": "Negative",
                "description": "Describes a negative sentiment or event"
            }
        ],
        "examples": [
            {
                "query": "I absolutely love this new restaurant! The food was amazing.",
                "classification": "Positive"
            },
            {
                "query": "This product is terrible. It broke after just one day.",
                "classification": "Negative"
            },
            {
                "query": "What a fantastic day! The weather is perfect.",
                "classification": "Positive"
            },
            {
                "query": "I'm really disappointed with the movie. The acting was poor.",
                "classification": "Negative"
            }
        ]
    }
}


def get_template(name: str) -> Dict[str, Any]:
    """
    Get a schema template by name

    Args:
        name: Template name

    Returns:
        Schema dictionary

    Raises:
        KeyError: If template name not found
    """
    if name not in TEMPLATES:
        raise KeyError(f"Template '{name}' not found. Available templates: {list(TEMPLATES.keys())}")
    return TEMPLATES[name].copy()


def get_template_names() -> List[str]:
    """
    Get list of available template names

    Returns:
        List of template names
    """
    return list(TEMPLATES.keys())
