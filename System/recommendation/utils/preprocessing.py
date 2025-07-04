"""
Data preprocessing utilities for recommendation algorithms.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Optional
import os

def prepare_audio_features(
    audio_features_df: pd.DataFrame,
    handle_missing: str = 'mean'
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Prepare audio features for recommendation.
    
    Args:
        audio_features_df: DataFrame containing audio features
        handle_missing: How to handle missing values ('mean', 'median', 'drop')
        
    Returns:
        Tuple containing:
            - Normalized audio feature matrix
            - List of song IDs
            - List of feature names
    """
    # Check for NaN values in each column
    nan_counts = audio_features_df.isna().sum()
    nan_columns = nan_counts[nan_counts > 0]
    
    if not nan_columns.empty:
        print("\nNaN counts in audio features:")
        print(nan_columns)
        
        # Handle missing values
        for col in nan_columns.index:
            if handle_missing == 'mean':
                value = audio_features_df[col].mean()
                print(f"Imputing {nan_counts[col]} missing values in {col} with mean: {value:.4f}")
            elif handle_missing == 'median':
                value = audio_features_df[col].median()
                print(f"Imputing {nan_counts[col]} missing values in {col} with median: {value:.4f}")
            elif handle_missing == 'drop':
                audio_features_df = audio_features_df.drop(columns=[col])
                print(f"Dropping column {col} with {nan_counts[col]} missing values")
                continue
            else:
                raise ValueError(f"Invalid handle_missing value: {handle_missing}")
            
            audio_features_df.loc[:, col] = audio_features_df[col].fillna(value)
    
    # Extract song IDs
    song_ids = audio_features_df.iloc[:, 0].tolist()
    
    # Extract just the numeric features, not the ID column
    feature_cols = [col for col in audio_features_df.columns 
                   if col != audio_features_df.columns[0] and 
                   audio_features_df[col].dtype in [np.float64, np.int64]]
    
    # Create feature matrix
    features = audio_features_df[feature_cols].values
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    return normalized_features, song_ids, feature_cols


def prepare_tag_features(tags_df: pd.DataFrame, song_ids: List[str] = None) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Prepare tag features for content-based recommendation.
    
    Args:
        tags_df: DataFrame containing tag data
        song_ids: Optional list of song IDs to filter by (if None, use all songs in tags_df)
        
    Returns:
        Tuple of (tag_features, song_ids, tag_columns)
    """
    # Identify metadata columns to exclude
    metadata_cols = ['id_dataset', 'id_spotify', 'url_spotify_preview', 'url_lastfm', 'artist', 'name']
    id_column = tags_df.columns[0]  # Assume first column is ID
    
    # If song_ids is provided, filter the tags DataFrame to only include those songs
    if song_ids is not None:
        print(f"Filtering tag features to {len(song_ids)} specified song IDs...")
        tags_df = tags_df[tags_df[id_column].isin(song_ids)].copy()
        print(f"After filtering: {len(tags_df)} songs with tag data")
    
    # Get song IDs from the filtered tags dataframe
    filtered_song_ids = tags_df[id_column].tolist()
    
    # Select only binary tag columns (0/1 values)
    tag_columns = []
    for col in tags_df.columns:
        if col not in metadata_cols and tags_df[col].isin([0, 1]).all():
            tag_columns.append(col)
    
    # Create tag features DataFrame with ID column included
    filtered_tags = tags_df[[id_column] + tag_columns].copy()
    
    # Ensure all songs in song_ids are included (add rows with zeros for missing songs)
    if song_ids is not None:
        missing_songs = set(song_ids) - set(filtered_song_ids)
        if missing_songs:
            print(f"Adding {len(missing_songs)} missing songs to tag features with zero values")
            missing_df = pd.DataFrame(0, index=range(len(missing_songs)), columns=tag_columns)
            missing_df[id_column] = list(missing_songs)
            filtered_tags = pd.concat([filtered_tags, missing_df], ignore_index=True)
            
            # Update filtered_song_ids to include the missing songs
            filtered_song_ids = song_ids.copy()
    
    # Sort by song_ids order
    if song_ids is not None:
        id_to_idx = {id: i for i, id in enumerate(song_ids)}
        filtered_tags['sort_idx'] = filtered_tags[id_column].map(id_to_idx)
        filtered_tags = filtered_tags.sort_values('sort_idx').drop('sort_idx', axis=1)
    
    # Extract just the tag features (without ID column)
    tag_features = filtered_tags[tag_columns].values.astype(np.float32)  # Use float32 to reduce memory
    
    return tag_features, filtered_song_ids, tag_columns


def align_interaction_song_ids(interactions_df, feature_song_ids, mapping_df=None, mapping_path=None):
    """
    Align song IDs in interactions with song IDs in features.
    
    Args:
        interactions_df: DataFrame containing user-song interactions
        feature_song_ids: List of song IDs in the feature matrix
        mapping_df: Optional DataFrame containing mapping between different song ID formats
        mapping_path: Optional path to a CSV file containing the mapping
        
    Returns:
        DataFrame with aligned song IDs
    """
    print("Aligning interaction song IDs with feature song IDs...")
    
    # Convert feature_song_ids to a set for faster lookup
    feature_song_ids_set = set(feature_song_ids)
    
    # Get unique song IDs from interactions
    interaction_song_ids = set(interactions_df['song_id'].unique())
    
    # Print diagnostic information
    print("Diagnostic - Song ID overlap:")
    print(f"Unique song IDs in interactions: {len(interaction_song_ids)}")
    print(f"Unique song IDs in features: {len(feature_song_ids_set)}")
    
    # Check overlap
    overlap = interaction_song_ids.intersection(feature_song_ids_set)
    print(f"Overlap (songs in both): {len(overlap)}")
    print(f"Percentage of interaction songs in features: {len(overlap) / len(interaction_song_ids) * 100:.2f}%")
    
    # If there's no overlap, try to use the mapping
    if len(overlap) == 0 and (mapping_df is not None or mapping_path is not None):
        print("No overlap between interaction and feature song IDs. Attempting to align...")
        
        # Load mapping if path is provided
        if mapping_df is None and mapping_path is not None:
            mapping_df = pd.read_csv(mapping_path)
            print(f"Loaded mapping table with {len(mapping_df)} entries")
        
        if mapping_df is not None:
            # Check if the mapping has the necessary columns
            if 'msd_song_id' in mapping_df.columns and 'lastfm_id' in mapping_df.columns:
                # Create a dictionary for mapping MSD IDs to Last.fm IDs
                msd_to_lastfm = dict(zip(mapping_df['msd_song_id'], mapping_df['lastfm_id']))
                
                # Create a dictionary for mapping Last.fm IDs to MSD IDs
                lastfm_to_msd = dict(zip(mapping_df['lastfm_id'], mapping_df['msd_song_id']))
                
                # Check which format the interactions are using
                sample_interaction_id = list(interaction_song_ids)[0] if interaction_song_ids else ""
                sample_feature_id = list(feature_song_ids_set)[0] if feature_song_ids_set else ""
                
                # Determine if interactions are using MSD IDs
                interactions_using_msd = sample_interaction_id.startswith('S') and len(sample_interaction_id) == 18
                
                # Determine if features are using MSD IDs
                features_using_msd = sample_feature_id.startswith('S') and len(sample_feature_id) == 18
                
                # Create a new DataFrame for the aligned interactions
                aligned_interactions = interactions_df.copy()
                
                # Map the song IDs based on the format
                if interactions_using_msd and not features_using_msd:
                    # Interactions use MSD IDs, features use Last.fm IDs
                    print("Interactions use MSD IDs, features use Last.fm IDs")
                    aligned_interactions['original_song_id'] = aligned_interactions['song_id']
                    aligned_interactions['song_id'] = aligned_interactions['song_id'].map(msd_to_lastfm)
                elif not interactions_using_msd and features_using_msd:
                    # Interactions use Last.fm IDs, features use MSD IDs
                    print("Interactions use Last.fm IDs, features use MSD IDs")
                    aligned_interactions['original_song_id'] = aligned_interactions['song_id']
                    aligned_interactions['song_id'] = aligned_interactions['song_id'].map(lastfm_to_msd)
                
                # Remove rows with NaN song_id (no mapping found)
                aligned_interactions = aligned_interactions.dropna(subset=['song_id'])
                
                # Check overlap after mapping
                new_interaction_song_ids = set(aligned_interactions['song_id'].unique())
                new_overlap = new_interaction_song_ids.intersection(feature_song_ids_set)
                print(f"After mapping: {len(new_overlap)} songs overlap")
                print(f"Percentage of interaction songs in features after mapping: {len(new_overlap) / len(new_interaction_song_ids) * 100:.2f}%")
                
                return aligned_interactions
            else:
                print(f"Warning: Mapping DataFrame does not have required columns. Available columns: {mapping_df.columns.tolist()}")
    
    # If there's already good overlap or mapping failed, return the original interactions
    return interactions_df
