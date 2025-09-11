#!/usr/bin/env python3

"""Test script to verify few_shot technique is working"""

try:
    from atap_llm_classifier.techniques import Technique
    print("✅ Successfully imported Technique")
    
    # Check all available techniques
    print("Available techniques:")
    for tech in Technique:
        print(f"  - {tech.value} ({tech.name})")
    
    # Test few_shot specifically
    print(f"\n🔍 Testing few_shot technique:")
    few_shot = Technique.FEW_SHOT
    print(f"  Enum value: {few_shot.value}")
    print(f"  Enum name: {few_shot.name}")
    
    # Test technique info
    info = few_shot.info
    print(f"  Name: {info.name}")
    print(f"  Description: {info.description}")
    
    # Test prompt template
    template = few_shot.prompt_template
    print(f"  Template loaded: {template is not None}")
    
    # Test prompt maker class
    prompt_maker_cls = few_shot.prompt_maker_cls
    print(f"  Prompt maker class: {prompt_maker_cls.__name__}")
    
    print("\n✅ All few_shot components working correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()