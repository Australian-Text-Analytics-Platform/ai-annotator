#!/usr/bin/env python3

import json
import sys
from pprint import pprint

# Load the schema file
try:
    with open('tests/example_user_schema2.json', 'r') as f:
        user_schema = json.load(f)
    print("✅ Schema file loaded successfully:")
    pprint(user_schema)
    print()
except Exception as e:
    print(f"❌ Error loading schema file: {e}")
    sys.exit(1)

# Test Pydantic validation
try:
    from atap_llm_classifier.techniques.schemas.cot import CoTUserSchema, CoTExample, CoTClass
    
    print("🔍 Testing direct Pydantic validation...")
    
    # Test individual components
    print("Testing CoTClass...")
    for cls_data in user_schema['classes']:
        cls_obj = CoTClass(**cls_data)
        print(f"  ✅ {cls_obj.name}: {cls_obj.description}")
    
    print("\nTesting CoTExample...")
    for ex_data in user_schema['examples']:
        print(f"  Input: {ex_data}")
        try:
            ex_obj = CoTExample(**ex_data)
            print(f"  ✅ Example created: query='{ex_obj.query[:30]}...', classification='{ex_obj.classification}', reason={ex_obj.reason}")
        except Exception as e:
            print(f"  ❌ Example failed: {e}")
    
    print("\nTesting full CoTUserSchema...")
    schema_obj = CoTUserSchema(**user_schema)
    print(f"✅ Full schema validation successful!")
    print(f"   Classes: {len(schema_obj.classes)}")
    print(f"   Examples: {len(schema_obj.examples)}")
    
except Exception as e:
    print(f"❌ Pydantic validation error: {e}")
    import traceback
    traceback.print_exc()

# Test the technique validation method
try:
    print("\n🔍 Testing technique validation method...")
    from atap_llm_classifier.techniques import Technique
    
    cot_technique = Technique.CHAIN_OF_THOUGHT
    is_valid = cot_technique.prompt_maker_cls.is_validate_user_schema(user_schema)
    print(f"Technique validation result: {is_valid}")
    
    if is_valid:
        print("✅ Schema validation passed!")
    else:
        print("❌ Schema validation failed!")
        
except Exception as e:
    print(f"❌ Technique validation error: {e}")
    import traceback
    traceback.print_exc()