import pandas as pd
import numpy as np
from .preprocess import load_and_sample_data

def harmonize_king_county(df):
    """Harmonize King County dataset"""
    harmonized = df.copy()
    
    # Rename columns to standard names
    harmonized.rename(columns={
        'sqft_living': 'area_sqm',
        'price': 'price_usd',
        'bedrooms': 'bedrooms',
        'bathrooms': 'bathrooms'
    }, inplace=True)
    
    # Convert sqft to sqm
    harmonized['area_sqm'] = harmonized['area_sqm'] * 0.092903
    
    # Add region identifier
    harmonized['region'] = 'US_PacificNW'
    
    return harmonized[['price_usd', 'area_sqm', 'bedrooms', 'bathrooms', 'region']]

# Similar functions for Madrid and Australian datasets