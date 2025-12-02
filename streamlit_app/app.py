"""
AI Annotator Streamlit Demo App

A minimal web interface for the AI Annotator's batch classification capabilities.
"""
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd
import httpx

from api_client import FastAPIClient
from utils import (
    parse_csv,
    validate_schema,
    validate_json,
    format_results_for_download,
    should_warn_large_batch,
    build_classification_request
)
from schema_templates import get_template, get_template_names


# Page configuration
st.set_page_config(
    page_title="AI Annotator Demo",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize all session state variables with defaults"""
    defaults = {
        # Configuration
        'provider': 'openai',
        'model': None,
        'technique': 'zero_shot',
        'temperature': 1.0,
        'top_p': 1.0,
        'llm_api_key': '',
        'ollama_endpoint': os.getenv('OLLAMA_ENDPOINT', 'http://127.0.0.1:11434'),

        # Reasoning parameters
        'enable_reasoning': False,
        'max_reasoning_chars': 150,
        'reasoning_effort': None,

        # Data
        'uploaded_file': None,
        'df': None,
        'selected_column': None,
        'texts': None,
        'large_batch_confirmed': False,

        # Schema
        'schema_template': 'Sentiment Analysis (Zero-shot)',
        'user_schema': None,
        'schema_json_str': '',
        'schema_valid': False,
        'schema_error': None,

        # Job
        'job_id': None,
        'job_status': None,
        'job_progress': None,
        'job_results': None,
        'job_errors': None,
        'job_cost': None,
        'polling_active': False,
        'last_poll_time': None,
        'job_start_time': None,

        # Cost estimate
        'cost_estimate': None,

        # UI state
        'confirm_cancel': False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def check_environment() -> Optional[FastAPIClient]:
    """
    Check environment and initialize API client

    Returns:
        FastAPIClient instance or None if SERVICE_API_KEY not set
    """
    try:
        client = FastAPIClient()
        return client
    except ValueError as e:
        st.error(f"L {str(e)}")
        st.info("Please set the SERVICE_API_KEY environment variable and restart the app.")
        st.stop()


def load_models(client: FastAPIClient, provider: str) -> List[str]:
    """
    Load available models for selected provider

    Args:
        client: FastAPI client
        provider: Provider name

    Returns:
        List of model names
    """
    try:
        import re
        models_data = client.get_models()
        provider_models = [
            m['name'] for m in models_data['models']
            if m['provider'] == provider
        ]

        # Filter out VertexAI models for Gemini provider
        # VertexAI models end in -XXX (e.g., gemini-1.5-flash-001)
        # Gemini API models don't have version suffixes
        if provider == "gemini":
            provider_models = [m for m in provider_models if not re.search(r'-\d{3}$', m)]

        return provider_models
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        return []


def render_sidebar(client: FastAPIClient):
    """Render sidebar configuration panel"""
    st.sidebar.title("Configuration")

    # Environment notice
    st.sidebar.info("SERVICE_API_KEY environment variable is configured ")

    # Provider selection
    st.sidebar.subheader("Provider Settings")
    providers = ["openai", "gemini", "anthropic", "ollama"]
    try:
        provider_index = providers.index(st.session_state.provider)
    except ValueError:
        provider_index = 0

    provider = st.sidebar.radio(
        "LLM Provider",
        providers,
        index=provider_index,
        key="provider_radio",
        help="Select your preferred LLM provider"
    )

    if provider != st.session_state.provider:
        st.session_state.provider = provider
        st.session_state.model = None  # Reset model when provider changes

    # Provider-specific inputs
    if provider == "openai":
        # Check if OpenAI API key is in environment
        env_openai_key = os.getenv("OPENAI_API_KEY", "")

        if env_openai_key:
            # Use key from .env and show confirmation
            st.session_state.llm_api_key = env_openai_key
            st.sidebar.success("✓ OpenAI API Key loaded from .env")
        else:
            # Show input field if not in .env
            llm_api_key = st.sidebar.text_input(
                "OpenAI API Key",
                type="password",
                value=st.session_state.llm_api_key,
                help="Enter your OpenAI API key"
            )
            st.session_state.llm_api_key = llm_api_key

    elif provider == "gemini":
        # Check if Gemini API key is in environment
        env_gemini_key = os.getenv("GEMINI_API_KEY", "")

        if env_gemini_key:
            # Use key from .env and show confirmation
            st.session_state.llm_api_key = env_gemini_key
            st.sidebar.success("✓ Gemini API Key loaded from .env")
        else:
            # Show input field if not in .env
            llm_api_key = st.sidebar.text_input(
                "Gemini API Key",
                type="password",
                value=st.session_state.llm_api_key,
                help="Enter your Google Gemini API key"
            )
            st.session_state.llm_api_key = llm_api_key

    elif provider == "anthropic":
        # Check if Anthropic API key is in environment
        env_anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

        if env_anthropic_key:
            # Use key from .env and show confirmation
            st.session_state.llm_api_key = env_anthropic_key
            st.sidebar.success("✓ Anthropic API Key loaded from .env")
        else:
            # Show input field if not in .env
            llm_api_key = st.sidebar.text_input(
                "Anthropic API Key",
                type="password",
                value=st.session_state.llm_api_key,
                help="Enter your Anthropic API key"
            )
            st.session_state.llm_api_key = llm_api_key

    elif provider == "ollama":
        # Check if Ollama endpoint is in environment
        env_ollama_endpoint = os.getenv("OLLAMA_ENDPOINT", "")

        if env_ollama_endpoint:
            # Use endpoint from .env and show confirmation
            st.session_state.ollama_endpoint = env_ollama_endpoint
            st.sidebar.success(f"✓ Ollama endpoint loaded from .env: {env_ollama_endpoint}")
        else:
            # Show input field if not in .env
            ollama_endpoint = st.sidebar.text_input(
                "Ollama Endpoint",
                value=st.session_state.ollama_endpoint,
                help="Ollama server endpoint URL"
            )
            st.session_state.ollama_endpoint = ollama_endpoint

    # Model selection
    st.sidebar.subheader("Model Settings")

    with st.spinner("Loading models..."):
        available_models = load_models(client, provider)

    if available_models:
        # Set default model if not set or if it's not in available models
        if st.session_state.model not in available_models:
            # Set provider-specific defaults
            if provider == "openai" and "gpt-4.1-mini" in available_models:
                st.session_state.model = "gpt-4.1-mini"
            elif provider == "gemini" and "gemini-2.5-flash" in available_models:
                st.session_state.model = "gemini-2.5-flash"
            elif provider == "anthropic" and "claude-4.5-haiku" in available_models:
                st.session_state.model = "claude-4.5-haiku"
            else:
                st.session_state.model = available_models[0]

        model = st.sidebar.selectbox(
            "Model",
            available_models,
            index=available_models.index(st.session_state.model),
            help="Select the LLM model to use"
        )
        st.session_state.model = model
    else:
        st.sidebar.warning("No models available for this provider")
        st.session_state.model = None

    # Technique selection
    technique = st.sidebar.selectbox(
        "Classification Technique",
        ["zero_shot", "few_shot"],
        index=0 if st.session_state.technique == "zero_shot" else 1,
        help="Classification technique to use"
    )
    st.session_state.technique = technique

    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Controls randomness (higher = more random)"
        )
        st.session_state.temperature = temperature

        top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.top_p,
            step=0.05,
            help="Nucleus sampling parameter"
        )
        st.session_state.top_p = top_p

        st.markdown("#### Reasoning Settings")

        # Enable reasoning toggle
        enable_reasoning = st.checkbox(
            "Enable Reasoning Output",
            value=st.session_state.enable_reasoning,
            help="Request the LLM to provide a brief explanation for each classification"
        )
        st.session_state.enable_reasoning = enable_reasoning

        # Max reasoning characters (only show if enable_reasoning is True)
        if enable_reasoning:
            max_reasoning_chars = st.slider(
                "Max Reasoning Characters",
                min_value=50,
                max_value=500,
                value=st.session_state.max_reasoning_chars,
                step=50,
                help="Maximum length of reasoning explanations"
            )
            st.session_state.max_reasoning_chars = max_reasoning_chars

        # Reasoning effort (for compatible models)
        reasoning_effort_options = ["None", "low", "medium", "high"]
        reasoning_effort_index = 0
        if st.session_state.reasoning_effort in ["low", "medium", "high"]:
            reasoning_effort_index = reasoning_effort_options.index(st.session_state.reasoning_effort)

        reasoning_effort_display = st.selectbox(
            "Reasoning Effort",
            options=reasoning_effort_options,
            index=reasoning_effort_index,
            help="Native reasoning mode for models that support it (leave as 'None' for models without native reasoning support)"
        )
        reasoning_effort = None if reasoning_effort_display == "None" else reasoning_effort_display.lower()
        st.session_state.reasoning_effort = reasoning_effort

        # Display info about reasoning modes
        if reasoning_effort:
            st.info("Native reasoning effort is only supported by some models and will be ignored for others.")

    # Help documentation for reasoning features
    with st.sidebar.expander("About Reasoning Features"):
        st.markdown("""
        ### Confidence Scores
        All classifications include a confidence score (0-1) indicating the model's certainty.

        ### Reasoning Output
        Enable "Reasoning Output" to request explanations for classifications.
        The model will provide a brief rationale for its decisions.

        ### Reasoning Effort
        For models that support native reasoning:
        - **Low**: Faster, basic reasoning
        - **Medium**: Balanced reasoning depth
        - **High**: Most thorough reasoning (slower, more expensive)

        Note: Reasoning effort only works with compatible models and will be ignored for models without native reasoning support.
        """)


def render_file_upload():
    """Render file upload section"""
    st.header("1. Upload CSV File")

    st.info("💡 Need test data? Use `tests/example_dataset.csv` from the repository.")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file containing texts to classify"
    )

    if uploaded_file is not None:
        try:
            df = parse_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.uploaded_file = uploaded_file

            # Column selector
            columns = df.columns.tolist()
            if columns:
                selected_column = st.selectbox(
                    "Select text column",
                    columns,
                    help="Choose the column containing text to classify"
                )
                st.session_state.selected_column = selected_column

                # Extract texts
                st.session_state.texts = df[selected_column].astype(str).tolist()

                # Preview
                st.subheader("Data Preview")
                st.dataframe(df[[selected_column]].head(), width='stretch')

                # Show batch size
                num_texts = len(st.session_state.texts)
                st.info(f"Total texts to classify: {num_texts}")

                # Large batch warning
                if should_warn_large_batch(num_texts):
                    st.warning(
                        f"⚠️ You are about to classify {num_texts} texts. "
                        "This may take a while and incur significant costs."
                    )
                    large_batch_confirmed = st.checkbox(
                        "I understand and want to proceed",
                        value=st.session_state.large_batch_confirmed
                    )
                    st.session_state.large_batch_confirmed = large_batch_confirmed
                else:
                    st.session_state.large_batch_confirmed = True

        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            st.session_state.df = None
            st.session_state.texts = None
    else:
        st.session_state.df = None
        st.session_state.texts = None
        st.session_state.large_batch_confirmed = False


def render_schema_configuration():
    """Render schema configuration section"""
    st.header("2. Configure Classification Schema")

    # Template selector
    template_names = get_template_names() + ["Custom"]
    template = st.selectbox(
        "Select a template",
        template_names,
        index=template_names.index(st.session_state.schema_template)
            if st.session_state.schema_template in template_names else 0,
        help="Choose a pre-defined schema or create a custom one"
    )

    # Load template if changed
    if template != st.session_state.schema_template:
        st.session_state.schema_template = template
        if template != "Custom":
            schema_dict = get_template(template)
            st.session_state.schema_json_str = json.dumps(schema_dict, indent=2)
        else:
            if not st.session_state.schema_json_str:
                st.session_state.schema_json_str = json.dumps({
                    "classes": [
                        {"name": "ClassName", "description": "Description"}
                    ]
                }, indent=2)

    # Initialize schema JSON string if empty
    if not st.session_state.schema_json_str and template != "Custom":
        schema_dict = get_template(template)
        st.session_state.schema_json_str = json.dumps(schema_dict, indent=2)

    # JSON editor
    schema_json_str = st.text_area(
        "Schema JSON",
        value=st.session_state.schema_json_str,
        height=300,
        help="Edit the classification schema in JSON format"
    )
    st.session_state.schema_json_str = schema_json_str

    # Validate schema
    is_valid_json, schema_dict, json_error = validate_json(schema_json_str)

    if is_valid_json:
        is_valid_schema, schema_error = validate_schema(schema_dict, st.session_state.technique)

        if is_valid_schema:
            st.success(" Schema is valid")
            st.session_state.user_schema = schema_dict
            st.session_state.schema_valid = True
            st.session_state.schema_error = None
        else:
            st.error(f"L Invalid schema: {schema_error}")
            st.session_state.schema_valid = False
            st.session_state.schema_error = schema_error
            st.session_state.user_schema = None
    else:
        st.error(f"L {json_error}")
        st.session_state.schema_valid = False
        st.session_state.schema_error = json_error
        st.session_state.user_schema = None


def render_job_submission(client: FastAPIClient):
    """Render job submission section"""
    st.header("3. Submit Classification Job")

    # Check if all required fields are ready
    can_submit = (
        st.session_state.texts is not None
        and st.session_state.user_schema is not None
        and st.session_state.schema_valid
        and st.session_state.model is not None
        and st.session_state.large_batch_confirmed
        and not st.session_state.polling_active
    )

    # Cost estimation
    col1, col2 = st.columns([1, 3])

    with col1:
        estimate_clicked = st.button(
            "Estimate Cost",
            disabled=not can_submit,
            help="Estimate the cost before submitting"
        )

    if estimate_clicked and can_submit:
        try:
            with st.spinner("Estimating cost..."):
                # Take first few texts as sample
                sample_texts = st.session_state.texts[:min(5, len(st.session_state.texts))]
                total_texts = len(st.session_state.texts)

                estimate_request = {
                    "texts": sample_texts,
                    "user_schema": st.session_state.user_schema,
                    "provider": st.session_state.provider,
                    "model": st.session_state.model,
                    "technique": st.session_state.technique,
                    "enable_reasoning": st.session_state.enable_reasoning,
                    "max_reasoning_chars": st.session_state.max_reasoning_chars
                }

                sample_estimate = client.estimate_cost(estimate_request)

                # Extrapolate to full dataset
                if sample_estimate and len(sample_texts) > 0:
                    scaling_factor = total_texts / len(sample_texts)

                    cost_estimate = {
                        "num_texts": total_texts,
                        "sample_size": len(sample_texts),
                        "input_tokens": int(sample_estimate.get("input_tokens", 0) * scaling_factor) if sample_estimate.get("input_tokens") else None,
                        "output_tokens": int(sample_estimate.get("output_tokens", 0) * scaling_factor) if sample_estimate.get("output_tokens") else None,
                        "reasoning_tokens": int(sample_estimate.get("reasoning_tokens", 0) * scaling_factor) if sample_estimate.get("reasoning_tokens") else None,
                        "estimated_tokens": int(sample_estimate.get("estimated_tokens", 0) * scaling_factor) if sample_estimate.get("estimated_tokens") else None,
                        "estimated_cost_usd": sample_estimate.get("estimated_cost_usd") * scaling_factor if sample_estimate.get("estimated_cost_usd") else None,
                        "input_cost_usd": sample_estimate.get("input_cost_usd") * scaling_factor if sample_estimate.get("input_cost_usd") else None,
                        "output_cost_usd": sample_estimate.get("output_cost_usd") * scaling_factor if sample_estimate.get("output_cost_usd") else None,
                        "reasoning_cost_usd": sample_estimate.get("reasoning_cost_usd") * scaling_factor if sample_estimate.get("reasoning_cost_usd") else None,
                        "input_cost_per_1m": sample_estimate.get("input_cost_per_1m"),
                        "output_cost_per_1m": sample_estimate.get("output_cost_per_1m"),
                        "warnings": sample_estimate.get("warnings", [])
                    }
                    st.session_state.cost_estimate = cost_estimate
                else:
                    st.session_state.cost_estimate = sample_estimate

        except Exception as e:
            st.error(f"Failed to estimate cost: {str(e)}")

    # Display cost estimate
    if st.session_state.cost_estimate:
        st.subheader("Cost Estimate")
        est = st.session_state.cost_estimate

        # Show sampling info if applicable
        if est.get('sample_size') and est.get('num_texts') and est['sample_size'] < est['num_texts']:
            st.info(f"📊 Estimated based on {est['sample_size']} sample texts, extrapolated to {est['num_texts']} total texts")

        # Show breakdown if available
        if est.get('input_tokens') and est.get('output_tokens'):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Input Tokens", f"{est['input_tokens']:,}")
            with col2:
                st.metric("Output Tokens", f"{est['output_tokens']:,}")
            with col3:
                st.metric("Total Tokens", f"{est['estimated_tokens']:,}")
            with col4:
                if est.get('estimated_cost_usd'):
                    st.metric("Estimated Cost", f"${est['estimated_cost_usd']:.4f}")
                else:
                    st.metric("Estimated Cost", "N/A")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Estimated Tokens", f"{est['estimated_tokens']:,}")
            with col2:
                if est.get('estimated_cost_usd'):
                    st.metric("Estimated Cost", f"${est['estimated_cost_usd']:.4f}")
                else:
                    st.metric("Estimated Cost", "N/A")
            with col3:
                st.metric("Sample Size", est['num_texts'])

        # Warning about reasoning tokens for reasoning models
        if st.session_state.model and 'gpt-4.1' in st.session_state.model.lower():
            st.warning(
                "⚠️ **Note on Reasoning Models:** Some models may use reasoning tokens which are NOT included in this estimate. "
                "Actual costs may be significantly higher due to additional reasoning tokens generated during inference."
            )

        if est.get('warnings'):
            for warning in est['warnings']:
                st.warning(warning)

    # Submit button
    st.subheader("Ready to Submit")

    if not can_submit:
        reasons = []
        if st.session_state.texts is None:
            reasons.append("Upload a CSV file")
        if st.session_state.user_schema is None or not st.session_state.schema_valid:
            reasons.append("Provide a valid schema")
        if st.session_state.model is None:
            reasons.append("Select a model")
        if not st.session_state.large_batch_confirmed:
            reasons.append("Confirm large batch processing")
        if st.session_state.polling_active:
            reasons.append("Wait for current job to complete")

        st.info(f"Complete the following to submit: {', '.join(reasons)}")

    submit_clicked = st.button(
        "Submit Classification Job",
        type="primary",
        disabled=not can_submit,
        help="Submit the batch classification job"
    )

    if submit_clicked and can_submit:
        try:
            with st.spinner("Submitting job..."):
                # Build request payload
                request_data = build_classification_request(
                    texts=st.session_state.texts,
                    user_schema=st.session_state.user_schema,
                    provider=st.session_state.provider,
                    model=st.session_state.model,
                    technique=st.session_state.technique,
                    temperature=st.session_state.temperature,
                    top_p=st.session_state.top_p,
                    llm_api_key=st.session_state.llm_api_key if st.session_state.provider in ["openai", "gemini", "anthropic"] else None,
                    llm_endpoint=st.session_state.ollama_endpoint if st.session_state.provider == "ollama" else None,
                    reasoning_effort=st.session_state.reasoning_effort,
                    enable_reasoning=st.session_state.enable_reasoning,
                    max_reasoning_chars=st.session_state.max_reasoning_chars
                )

                # Submit job
                response = client.submit_batch_job(request_data)

                # Store job info
                st.session_state.job_id = response['job_id']
                st.session_state.job_status = response['status']
                st.session_state.polling_active = True
                st.session_state.job_start_time = time.time()
                st.session_state.last_poll_time = time.time()

                st.success(f" Job submitted successfully! Job ID: {response['job_id']}")
                st.rerun()

        except httpx.HTTPStatusError as e:
            st.error(f"API Error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            st.error(f"Failed to submit job: {str(e)}")


def render_job_status_and_results(client: FastAPIClient):
    """Render job status and results section"""
    if st.session_state.job_id is None:
        return

    st.header("4. Job Status & Results")

    # Job ID
    st.text(f"Job ID: {st.session_state.job_id}")

    # Status badge and actions
    status = st.session_state.job_status
    if status:
        status_colors = {
            "pending": "🟡",
            "running": "🔵",
            "completed": "🟢",
            "failed": "🔴",
            "cancelled": "⚫"
        }
        st.markdown(f"**Status:** {status_colors.get(status, '⚪')} {status.upper()}")

    # Action buttons row
    col1, col2 = st.columns([1, 4])

    # Manual refresh button (always show when job exists)
    with col1:
        if st.button("🔄 Refresh", help="Manually refresh job status"):
            try:
                status_data = client.get_job_status(st.session_state.job_id)
                st.session_state.job_status = status_data['status']
                st.session_state.job_progress = status_data['progress']

                if status_data['status'] in ['completed', 'failed', 'cancelled']:
                    st.session_state.job_results = status_data.get('results')
                    st.session_state.job_errors = status_data.get('errors')
                    st.session_state.job_cost = status_data.get('cost')
                    st.session_state.polling_active = False

                st.success("Status refreshed!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to refresh: {str(e)}")

    # Cancel button (only show if pending or running)
    with col2:
        if status in ["pending", "running"]:
            if st.button("Cancel Job", type="secondary"):
                if st.session_state.confirm_cancel:
                    try:
                        result = client.cancel_job(st.session_state.job_id)
                        st.success(result['message'])
                        st.session_state.polling_active = False
                        st.session_state.job_status = 'cancelled'
                        st.session_state.confirm_cancel = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to cancel job: {str(e)}")
                else:
                    st.session_state.confirm_cancel = True
                    st.warning("Click 'Cancel Job' again to confirm")
                    time.sleep(0.5)
                    st.rerun()

    # Progress bar
    if st.session_state.job_progress:
        progress = st.session_state.job_progress
        st.progress(
            progress['percentage'] / 100.0,
            text=f"Progress: {progress['completed']}/{progress['total']} ({progress['percentage']:.1f}%)"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", progress['total'])
        with col2:
            st.metric("Completed", progress['completed'])
        with col3:
            st.metric("Failed", progress['failed'])

    # Results
    if st.session_state.job_status == "completed" and st.session_state.job_results:
        st.subheader("Results")

        # Create results DataFrame with new reasoning fields
        results_data = []
        for r in st.session_state.job_results:
            row = {
                'Index': r['index'],
                'Text': r['text'][:100] + '...' if len(r['text']) > 100 else r['text'],
                'Classification': r['classification']
            }
            # Add confidence if available
            if 'confidence' in r and r['confidence'] is not None:
                row['Confidence'] = round(r['confidence'], 2)
            # Add reasoning if available
            if 'reasoning' in r and r['reasoning']:
                row['Reasoning'] = r['reasoning']
            # Add reasoning_content if available
            if 'reasoning_content' in r and r['reasoning_content']:
                row['Reasoning (Native)'] = r['reasoning_content']
            results_data.append(row)

        results_df = pd.DataFrame(results_data)

        st.dataframe(results_df, use_container_width=True)

        # Show info about reasoning features used
        reasoning_info = []
        if 'Confidence' in results_df.columns:
            reasoning_info.append("✓ Confidence scores included")
        if 'Reasoning' in results_df.columns:
            reasoning_info.append("✓ Prompted reasoning included")
        if 'Reasoning (Native)' in results_df.columns:
            reasoning_info.append("✓ Native reasoning mode used")
        elif st.session_state.reasoning_effort:
            # User requested reasoning_effort but no reasoning_content in results
            reasoning_info.append(f"ℹ️ Reasoning effort '{st.session_state.reasoning_effort}' was set, but no native reasoning content was returned")

        if reasoning_info:
            st.info(" | ".join(reasoning_info))

        # Download button
        csv_data = format_results_for_download(st.session_state.job_results)
        st.download_button(
            label="Download Results CSV",
            data=csv_data,
            file_name=f"classification_results_{st.session_state.job_id}.csv",
            mime="text/csv"
        )

        # Cost information
        if st.session_state.job_cost:
            st.subheader("Cost Summary")
            cost = st.session_state.job_cost

            # Check if we have input/output breakdown
            has_breakdown = cost.get('input_tokens') is not None and cost.get('output_tokens') is not None

            if has_breakdown:
                # Show detailed token breakdown
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Input Tokens", f"{cost['input_tokens']:,}")
                with col2:
                    st.metric("Output Tokens", f"{cost['output_tokens']:,}")
                with col3:
                    if cost.get('reasoning_tokens'):
                        st.metric("Reasoning Tokens", f"{cost['reasoning_tokens']:,}")
                    else:
                        if cost.get('total_tokens'):
                            st.metric("Total Tokens", f"{cost['total_tokens']:,}")
                        else:
                            st.metric("Total Tokens", "N/A")
                with col4:
                    if cost.get('total_usd'):
                        st.metric("Total Cost", f"${cost['total_usd']:.4f}")
                    else:
                        st.metric("Total Cost", "N/A")

                # Detailed cost breakdown table
                if cost.get('input_cost_usd') or cost.get('output_cost_usd') or cost.get('reasoning_cost_usd'):
                    st.markdown("**Cost Breakdown:**")
                    breakdown_data = []

                    def format_cost(cost_val):
                        """Format cost value, handling very small amounts"""
                        if cost_val is None:
                            return "N/A"
                        elif cost_val == 0:
                            return "$0.0000"
                        elif cost_val < 0.00001:
                            return f"${cost_val:.2e}"  # Scientific notation
                        else:
                            return f"${cost_val:.4f}"

                    if cost.get('input_cost_usd') is not None:
                        breakdown_data.append({
                            "Type": "Input",
                            "Tokens": f"{cost.get('input_tokens', 0):,}",
                            "Cost": format_cost(cost['input_cost_usd'])
                        })

                    if cost.get('output_cost_usd') is not None:
                        breakdown_data.append({
                            "Type": "Output",
                            "Tokens": f"{cost.get('output_tokens', 0):,}",
                            "Cost": format_cost(cost['output_cost_usd'])
                        })

                    if cost.get('reasoning_cost_usd') is not None and cost.get('reasoning_tokens'):
                        breakdown_data.append({
                            "Type": "Reasoning",
                            "Tokens": f"{cost['reasoning_tokens']:,}",
                            "Cost": format_cost(cost['reasoning_cost_usd'])
                        })

                    if breakdown_data:
                        df_breakdown = pd.DataFrame(breakdown_data)
                        st.dataframe(df_breakdown, hide_index=True, width='content')

                    # Show note if all costs are zero
                    all_costs_zero = all(
                        cost.get(key, 0) == 0
                        for key in ['input_cost_usd', 'output_cost_usd', 'reasoning_cost_usd']
                        if cost.get(key) is not None
                    )
                    if all_costs_zero:
                        st.caption("Note: Cost breakdown shows $0 - pricing data may be unavailable for this model")

            else:
                # Simple display
                col1, col2 = st.columns(2)
                with col1:
                    if cost.get('total_usd'):
                        st.metric("Total Cost", f"${cost['total_usd']:.4f}")
                    else:
                        st.metric("Total Cost", "N/A")
                with col2:
                    if cost.get('total_tokens'):
                        st.metric("Total Tokens", f"{cost['total_tokens']:,}")
                    else:
                        st.metric("Total Tokens", "N/A")

            # Info about reasoning tokens if using reasoning model
            if st.session_state.model and 'gpt-4.1' in st.session_state.model.lower():
                if cost.get('reasoning_tokens'):
                    st.info(f"ℹ️ This job used {cost['reasoning_tokens']:,} reasoning tokens which are billed differently than standard output tokens.")
                else:
                    st.info("ℹ️ This model uses reasoning tokens which may increase costs beyond standard input/output tokens.")

    # Errors
    if st.session_state.job_status == "failed" and st.session_state.job_errors:
        st.subheader("Errors")
        st.error("Job failed with the following errors:")
        for error in st.session_state.job_errors:
            st.write(error)

    # Start new classification button (show when job is finished)
    if status in ["completed", "failed", "cancelled"]:
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Start New Classification", type="primary"):
                # Reset job-related state
                st.session_state.job_id = None
                st.session_state.job_status = None
                st.session_state.job_progress = None
                st.session_state.job_results = None
                st.session_state.job_errors = None
                st.session_state.job_cost = None
                st.session_state.polling_active = False
                st.session_state.last_poll_time = None
                st.session_state.job_start_time = None
                st.session_state.confirm_cancel = False
                st.session_state.cost_estimate = None
                st.success("Ready for a new classification job!")
                st.rerun()


def poll_job_status(client: FastAPIClient):
    """Poll job status and update session state"""
    if not st.session_state.polling_active or st.session_state.job_id is None:
        return

    # Check timeout (60 minutes)
    if st.session_state.job_start_time:
        elapsed = time.time() - st.session_state.job_start_time
        if elapsed > 3600:  # 60 minutes
            st.warning("⚠️ Polling timeout reached (60 minutes). Job may still be running.")
            st.session_state.polling_active = False
            return

    # Throttle polling to 2 seconds
    if st.session_state.last_poll_time:
        time_since_last_poll = time.time() - st.session_state.last_poll_time
        if time_since_last_poll < 2.0:
            time.sleep(2.0 - time_since_last_poll)

    try:
        status_data = client.get_job_status(st.session_state.job_id)

        # Update session state
        st.session_state.job_status = status_data['status']
        st.session_state.job_progress = status_data['progress']
        st.session_state.last_poll_time = time.time()

        # Check if job is complete
        if status_data['status'] in ['completed', 'failed', 'cancelled']:
            st.session_state.job_results = status_data.get('results')
            st.session_state.job_errors = status_data.get('errors')
            st.session_state.job_cost = status_data.get('cost')
            st.session_state.polling_active = False
            # Force final rerun to display results
            st.rerun()
        else:
            # Continue polling
            time.sleep(2.0)
            st.rerun()

    except Exception as e:
        st.error(f"Error polling job status: {str(e)}")
        st.session_state.polling_active = False


def main():
    """Main application"""
    # Initialize session state
    init_session_state()

    # Check environment and create API client
    client = check_environment()

    # Title
    st.title("🏷️ AI Annotator Demo")
    st.markdown("Batch text classification using LLM providers")

    # Sidebar
    render_sidebar(client)

    st.markdown("---")

    # Main sections
    render_file_upload()

    st.markdown("---")

    render_schema_configuration()

    st.markdown("---")

    render_job_submission(client)

    st.markdown("---")

    render_job_status_and_results(client)

    # Polling
    if st.session_state.polling_active:
        with st.spinner("Job in progress... (auto-refreshing)"):
            poll_job_status(client)


if __name__ == "__main__":
    main()
