#!/usr/bin/env python3

import json
import pandas as pd
import numpy as np

# Load calibrated rules
with open('artifacts/rules_calibrated/calibrated_rules.json', 'r') as f:
    rules = json.load(f)

# Load test data  
test_df = pd.read_pickle('artifacts/feature_selection/test_selected.pkl')
X_test = test_df.drop(columns=['success'])
y_test = test_df['success']

print("=== DEBUG: Module 8 vs Module 9 Scoring ===")
print(f"Test set size: {len(test_df)}")
print(f"Positive samples: {(y_test == 1).sum()}")
print(f"Negative samples: {(y_test == 0).sum()}")
print()

print("Rules:")
for i, rule in enumerate(rules):
    print(f"  {i+1}. {rule['feature']} (type: {rule['feature_type']}, threshold: {rule['threshold']}, precision: {rule['rule_precision']:.3f})")
print()

scoring_threshold = rules[0]['scoring_threshold']
print(f"Scoring threshold: {scoring_threshold:.6f}")
print()

# Apply each rule and compute scores (precision_weighted method)
weighted_scores = np.zeros(len(y_test))

for i, rule in enumerate(rules):
    feature = rule['feature']
    feature_type = rule['feature_type'] 
    threshold = rule['threshold']
    rule_precision = rule['rule_precision']
    
    # Apply rule
    if feature_type == 'continuous' and isinstance(threshold, bool):
        rule_mask = (X_test[feature] == threshold)
    elif feature_type in ('continuous', 'ordinal'):
        rule_mask = (X_test[feature] >= threshold)
    elif feature_type == 'binary':
        rule_mask = (X_test[feature] == True)
    elif feature_type == 'categorical':
        rule_mask = (X_test[feature] == threshold)
    
    # Add precision contribution  
    contribution = rule_precision
    weighted_scores += rule_mask * contribution
    
    # Debug stats
    predictions = rule_mask.sum()
    true_positives = (rule_mask & (y_test == 1)).sum()
    false_positives = (rule_mask & (y_test == 0)).sum()
    
    print(f"Rule {i+1} ({feature}):")
    print(f"  Predictions: {predictions}")
    print(f"  True Positives: {true_positives}")
    print(f"  False Positives: {false_positives}")
    print(f"  Contribution: {contribution:.6f}")
    print()

# Apply threshold
final_predictions = weighted_scores >= scoring_threshold
TP = int((final_predictions & (y_test == 1)).sum())
FP = int((final_predictions & (y_test == 0)).sum())
total_predictions = TP + FP

precision = TP / total_predictions if total_predictions > 0 else 0

print("=== FINAL RESULTS ===")
print(f"Weighted scores range: {weighted_scores.min():.6f} to {weighted_scores.max():.6f}")
print(f"Predictions above threshold: {total_predictions}")
print(f"True Positives: {TP}")
print(f"False Positives: {FP}") 
print(f"Precision: {precision:.6f}")
print()

print("=== EXPECTED FROM MODULE 8 ===")
print(f"Expected combined_support: {rules[0]['combined_support']}")
print(f"Expected combined_precision: {rules[0]['combined_precision']:.6f}")
print(f"Expected false positives: {rules[0].get('combined_false_positives', 'N/A')}")

# Show some examples of scores
print()
print("=== SCORE DISTRIBUTION ===")
unique_scores = sorted(set(weighted_scores[weighted_scores > 0]))
print(f"Unique positive scores: {unique_scores}")
print(f"Number of samples with each score:")
for score in unique_scores:
    count = (weighted_scores == score).sum()
    above_threshold = score >= scoring_threshold
    print(f"  {score:.6f}: {count} samples {'✓' if above_threshold else '✗'}") 