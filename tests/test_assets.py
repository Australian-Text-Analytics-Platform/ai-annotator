from atap_llm_classifier.techniques.techniques import Technique
from atap_llm_classifier.modifiers.modifiers import Modifier


def test_modifier_assets_all_exist_and_valid():
    for modifier in Modifier:
        props = modifier.get_properties()
    # pass if no exceptions raised.


def test_technique_assets_all_exist_and_valid():
    for technique in Technique:
        props = technique.properties()
    # pass if no exceptions raised.
