#!/usr/bin/env python3

import json
import pandas as pd

# Load data
test_df = pd.read_pickle('artifacts/feature_selection/test_selected.pkl')
X_test = test_df.drop(columns=['success'])
y_test = test_df['success']

# Load rules
with open('artifacts/rules_calibrated/calibrated_rules.json', 'r') as f:
    rules = json.load(f)

# Find the persona rule
persona_rule = None
for rule in rules:
    if rule['feature'] == 'persona':
        persona_rule = rule
        break

print("=== PERSONA RULE DEBUG ===")
print(f"Persona rule: {persona_rule}")
print()

# Check the actual threshold
threshold = persona_rule['threshold']
print(f"Threshold: {threshold}")
print(f"Threshold type: {type(threshold)}")
print()

# Check persona column
persona_values = X_test['persona']
print(f"Persona column type: {persona_values.dtype}")
print(f"Sample persona values:")
for i, val in enumerate(persona_values.head(10)):
    print(f"  {i}: {repr(val)}")
print()

# Test different matching approaches
print("=== TESTING RULE APPLICATION ===")

# 1. Direct equality (what Module 9 is currently doing)
direct_match = (persona_values == threshold)
print(f"1. Direct equality match (persona == threshold):")
print(f"   Matches: {direct_match.sum()}")
print(f"   Matching values: {persona_values[direct_match].unique()}")
print()

# 2. Check if threshold is string representation of a list
if isinstance(threshold, str):
    str_match = (persona_values == threshold)
    print(f"2. String match (persona == '{threshold}'):")
    print(f"   Matches: {str_match.sum()}")
    print()

# 3. Check if we should parse the threshold as a list
if isinstance(threshold, list):
    # Convert list to string representation like the data
    threshold_str = str(threshold)
    str_repr_match = (persona_values == threshold_str)
    print(f"3. List-to-string match (persona == '{threshold_str}'):")
    print(f"   Matches: {str_repr_match.sum()}")
    print()

# 4. Check what the actual matching value should be
print("=== EXPECTED MATCH ===")
target_str = "['L2_1', 'L3_3', 'L3_6']"
correct_match = (persona_values == target_str)
print(f"Correct match (persona == \"{target_str}\"):")
print(f"   Matches: {correct_match.sum()}")
if correct_match.sum() > 0:
    matching_indices = correct_match[correct_match].index.tolist()
    print(f"   Matching indices: {matching_indices}")
    for idx in matching_indices:
        success_val = y_test.iloc[idx]
        print(f"   Index {idx}: success = {success_val}")
print()

# Check why Module 9 might be matching so many
print("=== DIAGNOSING THE 766 MATCHES ===")
# Maybe there's a boolean conversion issue?
bool_threshold = bool(threshold)
print(f"bool(threshold) = {bool_threshold}")

# Maybe it's treating it as "any non-empty list"?
non_empty_match = persona_values.notna() & (persona_values != "[]") & (persona_values != "['']")
print(f"Non-empty persona values: {non_empty_match.sum()}") 