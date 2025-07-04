"""Utilities for displaying recommendation details."""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def print_recommendation_details(
    recommendations: pd.DataFrame,
    header: str = "RECOMMENDATIONS",
    max_items: int = 20,
    columns_to_display: list = None
) -> None:
    """Print detailed information about recommendations in a table format.
    
    Args:
        recommendations: DataFrame containing recommendations
        header: Custom header for the table
        max_items: Maximum number of items to print
        columns_to_display: List of column names to display (e.g., ['song_id', 'title', 'artist_name', 'top_genre', 'language'])
    """
    if recommendations.empty:
        logger.warning("No recommendations to display for %s", header)
        print(f"\n=== {header} ===\nNo recommendations to display.")
        return
    
    # Default columns if none specified
    if columns_to_display is None:
        columns_to_display = []
        for col in ['song_id', 'title', 'song_title']:
            if col in recommendations.columns:
                columns_to_display.append(col)
                break
        for col in ['artist_name', 'artist']:
            if col in recommendations.columns:
                columns_to_display.append(col)
                break
        if 'top_genre' in recommendations.columns:
            columns_to_display.append('top_genre')
        if 'language' in recommendations.columns:
            columns_to_display.append('language')
        if 'artist_tier' in recommendations.columns:
            columns_to_display.append('artist_tier')
        if 'score' in recommendations.columns or 'similarity' in recommendations.columns:
            columns_to_display.append('score' if 'score' in recommendations.columns else 'similarity')
    
    # Validate requested columns
    valid_columns = [col for col in columns_to_display if col in recommendations.columns]
    if not valid_columns:
        logger.warning("No valid columns found in recommendations for %s. Available columns: %s", header, list(recommendations.columns))
        print(f"\n=== {header} ===\nNo valid metadata columns found to display.")
        return
    
    # Build table
    print("\n" + "=" * 80)
    print(f"{header}")
    print("=" * 80)
    
    # Define column widths and headers
    col_widths = {
        'song_id': 36,
        'title': 30,
        'song_title': 30,
        'artist_name': 20,
        'artist': 20,
        'top_genre': 15,
        'language': 10
    }
    col_headers = {
        'song_id': 'Song ID',
        'title': 'Title',
        'song_title': 'Title',
        'artist_name': 'Artist',
        'artist': 'Artist',
        'top_genre': 'Genre',
        'language': 'Language'
    }
    
    # Build header row
    table_header = ""
    for col in valid_columns:
        header = col_headers.get(col, col)
        width = col_widths.get(col, 20)
        table_header += f"{header:<{width}} | "
    print(table_header.rstrip(" | "))
    print("-" * len(table_header.rstrip(" | ")))
    
    # Print rows
    for i, (_, row) in enumerate(recommendations.iterrows()):
        if i >= max_items:
            break
        line = ""
        for col in valid_columns:
            width = col_widths.get(col, 20)
            value = str(row[col])[:width-2] if pd.notna(row[col]) else "Unknown"
            line += f"{value:<{width}} | "
        print(line.rstrip(" | "))
    
    print("=" * 80)