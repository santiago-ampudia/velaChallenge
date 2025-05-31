"""
Data cleaning parameters for Module 1.

This module defines all parameters used in data cleaning, including:
- Column type mappings based on feature documentation
- Imputation strategies
- Data type casting rules
"""

import os

# File paths (relative to project root)
INPUT_FILE = "raw_data/features_engineered.csv"
OUTPUT_DIR = "artifacts/prepared_data"
OUTPUT_FILE = "full_cleaned.pkl"
OUTPUT_CSV_FILE = "full_cleaned.csv"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Column type mapping based on feature documentation
COLUMN_TYPES = {
    # Identifiers (categorical)
    'founder_uuid': 'categorical',
    'name': 'categorical', 
    'org_name': 'categorical',
    
    # Education features
    'education_level': 'ordinal',  # 0-3 (associate to doctorate)
    'education_institution': 'ordinal',  # 0-4 (unknown to top 20)
    'education_field_of_study': 'ordinal',  # 0-3 (other to STEM)
    'education_international_experience': 'binary',
    'education_publications_and_research': 'binary',
    'education_extracurricular_involvement': 'binary',
    'education_awards_and_honors': 'binary',
    
    # Career metrics (continuous)
    'number_of_roles': 'continuous',
    'number_of_companies': 'continuous',
    'industry_achievements': 'continuous',
    
    # Experience flags (binary)
    'big_company_experience': 'binary',
    'nasdaq_company_experience': 'binary',
    'big_tech_experience': 'binary',
    'google_experience': 'binary',
    'facebook_meta_experience': 'binary',
    'microsoft_experience': 'binary',
    'amazon_experience': 'binary',
    'apple_experience': 'binary',
    'career_growth': 'binary',
    'moving_around': 'binary',
    'international_work_experience': 'binary',
    'worked_at_military': 'binary',
    'worked_at_consultancy': 'ordinal',  # 0-3 (none to top-tier)
    'worked_at_bank': 'ordinal',  # 0-3 (none to top-tier)
    'patents_inventions': 'binary',
    'technical_skills': 'binary',
    'technical_publications': 'binary',
    'technical_leadership_roles': 'binary',
    
    # Leadership levels (ordinal)
    'big_tech_position': 'ordinal',  # 0-5 (non-bigtech to researcher)
    'big_leadership': 'ordinal',  # 0-3 (none to C-level)
    'nasdaq_leadership': 'ordinal',  # 0-3 (none to C-level)
    'bigtech_leadership': 'ordinal',  # 0-3 (none to C-level)
    'number_of_leadership_roles': 'ordinal',  # 0-2 (none to multiple)
    'being_lead_of_nonprofits': 'binary',
    
    # Startup experience
    'startup_experience': 'binary',
    'previous_startup_funding_experience_as_ceo': 'binary',
    'previous_startup_funding_experience_as_nonceo': 'binary',
    'ceo_experience': 'binary',
    'investor_quality_prior_startup': 'ordinal',  # Quality levels
    'VC_experience': 'binary',
    'tier_1_VC_experience': 'binary',
    'angel_experience': 'binary',
    'quant_experience': 'binary',
    'board_advisor_roles': 'binary',
    
    # Media and influence (continuous)
    'press_media_coverage_count': 'continuous',
    'significant_press_media_coverage': 'binary',
    'speaker_influence': 'binary',
    
    # Personal attributes
    'professional_athlete': 'binary',
    'languages': 'categorical',  # Can be converted to count later
    'childhood_entrepreneurship': 'binary',
    'ten_thousand_hours_of_mastery': 'binary',
    'competitions': 'binary',
    
    # Personality scores (ordinal 0-2)
    'extroversion': 'ordinal',
    'perseverance': 'ordinal', 
    'risk_tolerance': 'ordinal',
    'vision': 'ordinal',
    'adaptability': 'ordinal',
    'emotional_intelligence': 'ordinal',
    'personal_branding': 'ordinal',
    
    # Derived features
    'persona': 'categorical',  # L0, L1, L2_X, L3_X
    'founder_experience': 'ordinal',
    'acquisition_experience': 'binary',
    'acquirer_bigtech': 'binary',
    'ipo_experience': 'binary',
    
    # Years of experience (continuous)
    'yoe': 'continuous',
    
    # Target variable
    'success': 'binary'
}

# Imputation strategies by column type
IMPUTATION_STRATEGIES = {
    'continuous': 'median',
    'ordinal': 'median', 
    'binary': 'mode',
    'categorical': 'empty_string'
}

# Data type casting by column type (after imputation)
DTYPE_CASTING = {
    'continuous': 'float32',
    'ordinal': 'int8',
    'binary': 'bool',
    'categorical': 'category'
}

# Columns that should be excluded from missing value flag creation
# (typically identifiers that we expect to always be present)
EXCLUDE_FROM_MISSING_FLAGS = [
    'founder_uuid',
    'name', 
    'org_name'
]

# Validation rules for continuous variables
CONTINUOUS_VALIDATION_RULES = {
    'number_of_roles': {'min_valid': 0},
    'number_of_companies': {'min_valid': 0},
    'industry_achievements': {'min_valid': 0},
    'press_media_coverage_count': {'min_valid': 0},
    'yoe': {'min_valid': 0}
} 