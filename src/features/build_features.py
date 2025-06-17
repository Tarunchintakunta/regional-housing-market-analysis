import pandas as pd
import numpy as np

def create_price_per_sqm(df):
    """Create price per square meter feature"""
    df['price_per_sqm'] = df['price_usd'] / df['area_sqm']
    return df

def create_room_features(df):
    """Create features based on room counts"""
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['bedroom_to_bathroom_ratio'] = df['bedrooms'] / df['bathrooms'].replace(0, 0.5)
    return df

def create_location_features(df, region):
    """Create region-specific location features"""
    if region == 'US_PacificNW':
        # King County specific location features
        pass
    elif region == 'EU_Madrid':
        # Madrid specific location features
        pass
    elif region == 'AU_Housing':
        # Australian specific location features
        pass
    return df