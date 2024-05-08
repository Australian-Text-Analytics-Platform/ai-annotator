from atap_llm_classifier.techniques.techniques import Technique
from atap_llm_classifier.modifiers.modifiers import Modifier


def test_modifier_assets():
    for modifier in Modifier:
        ctx = modifier.get_context()
    # pass if no exceptions raised.


def test_technique_assets():
    for technique in Technique:
        ctx = technique.get_context()
    # pass if no exceptions raised.
