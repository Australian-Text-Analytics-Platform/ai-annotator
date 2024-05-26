from atap_llm_classifier.modifiers.modifiers import Modifier
from atap_llm_classifier.formatter.models import OutputFormat
from atap_llm_classifier.providers import LLMProvider
from atap_llm_classifier.techniques.techniques import Technique
from atap_llm_classifier.views.props import ViewProp


def test_modifier_props_all_exist_and_valid():
    for modifier in Modifier:
        props = modifier.properties
    # pass if no exceptions raised.


def test_technique_props_all_exist_and_valid():
    for technique in Technique:
        props = technique.info
    # pass if no exceptions raised.


def test_prompt_templates_all_exist_and_valid():
    for technique in Technique:
        prompt_template = technique.prompt_template
    # pass if no exceptions raised.


def test_provider_props_all_exist_and_valid():
    for provider in LLMProvider:
        props = provider.properties
    # pass if no exceptions raised.


def test_output_formatter_infos_is_valid():
    info = OutputFormat.infos()
    # pass if no exceptions raised.


def test_output_formatter_templates_all_exist_and_valid():
    for out_format in OutputFormat:
        template = out_format.template


def test_view_props_all_exist_and_valid():
    for viewprop in ViewProp:
        prop = viewprop.properties
    # pass if no exceptions raised.
