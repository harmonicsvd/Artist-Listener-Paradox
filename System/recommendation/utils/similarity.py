"""Similarity computation utilities for recommendation systems."""

from typing import Dict, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize
import logging

logger = logging.getLogger(__name__)

def compute_content_similarity(
    features: np.ndarray,
    chunk_size: int = 500,
    metric: str = 'cosine'
) -> np.ndarray:
    """Compute pairwise item similarity matrix efficiently.
    
    Args:
        features: Feature matrix of shape (n_items, n_features)
        chunk_size: Process items in chunks for memory efficiency
        metric: Similarity metric ('cosine', 'pearson', 'euclidean')
        
    Returns:
        Similarity matrix of shape (n_items, n_items)
        
    Raises:
        ValueError: If unsupported metric or invalid features
    """
    if features.ndim != 2:
        raise ValueError("Features must be a 2D array")
    
    logger.info("Computing similarity matrix with %s metric", metric)
    features_norm = normalize(features)
    
    if metric == 'cosine':
        sim_matrix = cosine_similarity(features_norm)
    elif metric == 'pearson':
        features_centered = features_norm - features_norm.mean(axis=1, keepdims=True)
        sim_matrix = cosine_similarity(features_centered)
    elif metric == 'euclidean':
        distances = euclidean_distances(features_norm)
        sim_matrix = 1 / (1 + distances)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Choose from ['cosine', 'pearson', 'euclidean']")
    
    logger.info("Similarity matrix computed for %d items", len(features))
    return sim_matrix

def get_similar_items(
    item_id: str,
    similarity_matrix: np.ndarray,
    item_ids: List[str],
    n: int = 10,
    exclude_self: bool = True
) -> Dict[str, float]:
    """Retrieve top-n most similar items for a given item.
    
    Args:
        item_id: Target item ID
        similarity_matrix: Precomputed similarity matrix
        item_ids: Ordered list of item IDs
        n: Number of similar items
        exclude_self: Exclude the query item
        
    Returns:
        Dictionary of {item_id: similarity_score}
        
    Raises:
        ValueError: If item_id not found
    """
    try:
        idx = item_ids.index(item_id)
    except ValueError:
        raise ValueError(f"Item {item_id} not found in item_ids")
    
    similarities = similarity_matrix[idx]
    ranked = sorted(zip(item_ids, similarities), key=lambda x: -x[1])
    if exclude_self:
        ranked = [x for x in ranked if x[0] != item_id]
    
    logger.info("Retrieved %d similar items for item %s", min(n, len(ranked)), item_id)
    return dict(ranked[:n])