import time
import pandas as pd
import numpy as np
import sys
import os
import logging
from functools import partial
from sklearn.model_selection import train_test_split
import optuna
import pickle
import tensorflow as tf # Import TensorFlow for TensorBoard

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Imports
from System.recommendation.recommendation_system import RecommendationSystem
from System.recommendation.content_based import ContentBasedRecommender
from System.recommendation.collaborative_filtering import CollaborativeFilteringRecommender
from System.recommendation.matrix_factorization import MatrixFactorizationRecommender
from System.recommendation.hybrid_recommender import HybridRecommender
from System.recommendation.evaluation import RecommendationEvaluator
from System.recommendation.cf_evaluation import CFRecommendationEvaluator
from System.recommendation.utils.display import print_recommendation_details
from System.recommendation.utils.mappings import create_genre_mapping, create_song_to_tier_mapping, create_language_mapping, create_artist_metric_mappings, compute_related_genres
from System.recommendation.objective_loss import ObjectiveLossCalculator
from System.recommendation.utils.recommendation_analysis import compare_recommendation_approaches
from System.recommendation.utils.user_analysis import analyze_user_listening_history
from optimisation import optimize_tier_weights

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_optuna_weights(db_path, study_name):
    """Load the best weights from an Optuna .db file."""
    try:
        storage = f"sqlite:///{db_path}"
        study = optuna.load_study(study_name=study_name, storage=storage)
        best_trial = study.best_trial
        best_weights = best_trial.user_attrs.get('best_weights_for_trial', best_trial.params)
        logger.info(f"Loaded best weights from {db_path}: {best_weights}")
        return best_weights, best_trial.value
    except Exception as e:
        logger.error(f"Failed to load weights from {db_path}: {str(e)}")
        return None, None

def split_data(interactions, song_metadata, test_size=0.5, random_state=42, min_interactions=6, max_users=None, load_saved_data=False, saved_data_dir=None):
    """Split interactions into train and test sets by user, ensuring min_interactions and focused genres/languages if possible."""
    if saved_data_dir and max_users:
        train_split_path = os.path.join(saved_data_dir, f"train_interactions_u{max_users}.csv")
        test_split_path = os.path.join(saved_data_dir, f"test_interactions_u{max_users}.csv")
    else:
        train_split_path = None
        test_split_path = None

    if load_saved_data and train_split_path and test_split_path and os.path.exists(train_split_path) and os.path.exists(test_split_path):
        logger.info(f"Loading saved data splits for {max_users} users from {saved_data_dir}")
        train_interactions = pd.read_csv(train_split_path)
        test_interactions = pd.read_csv(test_split_path)
        logger.info(f"Loaded train interactions: {len(train_interactions)}, test interactions: {len(test_interactions)}")
        return train_interactions, test_interactions

    user_counts = interactions['user_id'].value_counts()
    valid_users_initial = user_counts[user_counts >= min_interactions].index
    
    filtered_interactions = interactions[interactions['user_id'].isin(valid_users_initial)].copy()

    logger.info(f"Total users: {len(user_counts)}, Users with >={min_interactions} interactions: {len(valid_users_initial)}")
    logger.info(f"Columns in interactions: {list(filtered_interactions.columns)}")

    genre_column = 'top_genre'
    if genre_column not in song_metadata.columns:
        logger.error(f"Required genre column 'top_genre' not found in song_metadata. Available columns: {list(song_metadata.columns)}")
        genre_column = 'genre_fallback'
        song_metadata['genre_fallback'] = 'unknown_genre'
    else:
        logger.info(f"Found genre column 'top_genre' in song_metadata")

    if genre_column not in filtered_interactions.columns and 'song_id' in filtered_interactions.columns:
        logger.info(f"Merging {genre_column} from song_metadata into interactions")
        filtered_interactions = filtered_interactions.merge(
            song_metadata[['song_id', genre_column]],
            on='song_id',
            how='left'
        )
        filtered_interactions[genre_column] = filtered_interactions[genre_column].fillna('unknown_genre')

    language_column = 'language'
    if language_column not in song_metadata.columns:
        logger.warning(f"Language column 'language' not found in song_metadata. Available columns: {list(song_metadata.columns)}. Using fallback.")
        language_column = 'language_fallback'
        song_metadata['language_fallback'] = 'unknown_language'
    else:
        logger.info(f"Found language column 'language' in song_metadata")

    if language_column not in filtered_interactions.columns and 'song_id' in filtered_interactions.columns:
        logger.info(f"Merging {language_column} from song_metadata into interactions")
        filtered_interactions = filtered_interactions.merge(
            song_metadata[['song_id', language_column]],
            on='song_id',
            how='left'
        )
        filtered_interactions[language_column] = filtered_interactions[language_column].fillna('unknown_language')

    valid_users_final = valid_users_initial
    """
    if genre_column:
        user_genres = filtered_interactions.groupby('user_id')[genre_column].nunique()
        valid_users_genre_filtered = user_genres[user_genres <= 2].index
        valid_users_final = valid_users_initial.intersection(valid_users_genre_filtered)
        filtered_interactions = filtered_interactions[filtered_interactions['user_id'].isin(valid_users_final)].copy()
        logger.info(f"Users with >={min_interactions} interactions and <=2 genres (final count): {len(valid_users_final)}")
    else:
        logger.info(f"Users with >={min_interactions} interactions (no genre filter applied): {len(valid_users_final)}")
    """
    if len(valid_users_final) == 0:
        logger.error("No users with sufficient interactions after all filters. Cannot split data.")
        return pd.DataFrame(), pd.DataFrame()

    train_interactions = []
    test_interactions = []
    for user_id in valid_users_final:
        user_data = filtered_interactions[filtered_interactions['user_id'] == user_id]
        
        if len(user_data) < 2:
            logger.warning(f"User {user_id} has insufficient interactions ({len(user_data)}) for splitting. Skipping.")
            continue
        
        current_test_size = test_size
        if len(user_data) * test_size < 1:
            current_test_size = 1.0 / len(user_data)
            if current_test_size > 0.5:
                current_test_size = 0.5
            logger.warning(f"Adjusting test_size for user {user_id} due to small interaction count. New test_size: {current_test_size:.2f}")

        train, test = train_test_split(
            user_data,
            test_size=current_test_size,
            random_state=42,
        )
        train_interactions.append(train)
        test_interactions.append(test)

    if not train_interactions or not test_interactions:
        logger.error("No data left for train/test sets after splitting by user based on filters. Returning empty DataFrames.")
        return pd.DataFrame(), pd.DataFrame()

    train_interactions = pd.concat(train_interactions, ignore_index=True)
    test_interactions = pd.concat(test_interactions, ignore_index=True)

    if saved_data_dir and train_split_path and test_split_path:
        os.makedirs(saved_data_dir, exist_ok=True)
        train_interactions.to_csv(train_split_path, index=False)
        test_interactions.to_csv(test_split_path, index=False)
        logger.info(f"Saved train interactions to {train_split_path}, test interactions to {test_split_path}")

    logger.info(f"Final Train interactions: {len(train_interactions)}, Final Test interactions: {len(test_interactions)}")
    return train_interactions, test_interactions

def check_song_id_consistency(interactions, song_metadata, song_features):
    """Ensures consistency of song_ids across interactions, song_metadata, and song_features."""
    metadata_songs = set(song_metadata['song_id'])
    feature_songs = set(song_features['song_id'])
    
    consistent_item_ids = metadata_songs.intersection(feature_songs)

    if len(metadata_songs) != len(consistent_item_ids):
        removed_count = len(metadata_songs) - len(consistent_item_ids)
        logger.warning(f"Removed {removed_count} song_ids from song_metadata because they lack corresponding features.")
    if len(feature_songs) != len(consistent_item_ids):
        removed_count = len(feature_songs) - len(consistent_item_ids)
        logger.warning(f"Removed {removed_count} song_ids from song_features because they lack corresponding metadata.")

    filtered_song_metadata = song_metadata[song_metadata['song_id'].isin(consistent_item_ids)].copy()
    filtered_song_features = song_features[song_features['song_id'].isin(consistent_item_ids)].copy()
    interaction_songs_before_filter = set(interactions['song_id'])
    filtered_interactions = interactions[interactions['song_id'].isin(consistent_item_ids)].copy()

    if len(interaction_songs_before_filter) != len(set(filtered_interactions['song_id'])):
        removed_count = len(interaction_songs_before_filter) - len(set(filtered_interactions['song_id']))
        logger.warning(f"Removed {removed_count} song_ids from interactions because they lack metadata/features.")
    
    logger.info(f"Final consistent dataset for items has {len(consistent_item_ids)} unique song_ids.")
    logger.info(f"Final interactions contain {len(filtered_interactions['song_id'].unique())} unique song_ids.")

    return filtered_interactions, filtered_song_metadata, filtered_song_features

def format_metrics(metrics: dict, k_values: list, loss: dict = None) -> None:
    """Format and print evaluation metrics and objective loss."""
    print("\n" + "="*50)
    print("EVALUATION RESULTS AND OBJECTIVE LOSS")
    print("="*50)
    header = f"{'Metric':<25} " + " ".join(f"k={k:<10}" for k in k_values)
    print(header)
    print("-" * (25 + len(k_values) * 11))
    for metric in [
        'Precision', 'Genre Precision', 'Language Precision',
        'Recall', 'NDCG', 'Hit Rate', 'Emerging Artist Hit Rate',
        'Coverage (%)', 'Diversity', 'Novelty', 'Emerging Artist Exposure Index'
    ]:
        values = metrics.get(metric, {})
        row = f"{metric:<25} " + " ".join(
            f"{values.get(k, 0.0):<10.4f}" if metric != 'Coverage (%)'
            else f"{values.get(k, 0.0):<10.2f}%" for k in k_values
        )
        print(row)
    if loss:
        print("-" * (25 + len(k_values) * 11))
        loss_row = f"{'Objective Loss':<25} " + " ".join(
            f"{loss.get(k, 0.0):<10.4f}" for k in k_values
        )
        print(loss_row)

def initialize_shared_resources(config):
    """
    Initialize shared resources and mappings.
    This function handles loading or computing initial data and mappings.
    """
    logger.info("Initializing resources")
    
    mappings_path = os.path.join(config['saved_data_dir'], f"mappings_u{config['max_users']}.pkl") if config.get('saved_data_dir') else None

    system = RecommendationSystem(data_dir=config['data_path'])
    system.load_data(max_users=config['max_users'])
    
    user_interactions_raw = system.data_manager.user_loader.interactions
    song_metadata_raw = system.data_manager.song_loader.song_metadata
    song_features_raw = system.data_manager.song_loader.song_features
    artist_identification_raw = system.data_manager.artist_identification

    if song_metadata_raw.empty or user_interactions_raw.empty or song_features_raw.empty:
        logger.error("No data loaded from initial DataManager setup.")
        raise ValueError("Failed to load data")

    full_user_interactions, full_song_metadata, full_song_features = check_song_id_consistency(
        user_interactions_raw, song_metadata_raw, song_features_raw
    )
    system.data_manager.user_loader.interactions = full_user_interactions
    system.data_manager.song_loader.song_metadata = full_song_metadata
    system.data_manager.song_loader.song_features = full_song_features

    train_interactions, test_interactions = split_data(
        full_user_interactions,
        full_song_metadata,
        test_size=0.5,
        random_state=42,
        min_interactions=config.get('min_interactions', 6),
        max_users=config['max_users'],
        load_saved_data=config['load_saved_data'],
        saved_data_dir=config['saved_data_dir']
    )
    logger.info(f"Data split into Train: {len(train_interactions)} interactions, Test: {len(test_interactions)} interactions.")
    logger.info(f"Data diversity: {len(train_interactions['user_id'].unique())} unique users in train, {train_interactions['song_id'].nunique()} unique songs in train")

    logger.info(f"Columns in train_interactions after split_data: {list(train_interactions.columns)}")

    song_to_tier_map = {}
    song_to_pop_map = {}
    song_to_fam_map = {}
    song_to_genres_map = {}
    song_to_language_map = {}
    related_genres_map = {}
    song_to_artist_map = {}

    if config['load_saved_data'] and mappings_path and os.path.exists(mappings_path):
        logger.info(f"Attempting to load saved mappings for {config['max_users']} users from {mappings_path}")
        try:
            with open(mappings_path, 'rb') as f:
                saved_mappings = pickle.load(f)
            song_to_tier_map = saved_mappings['song_to_tier_map']
            song_to_pop_map = saved_mappings['song_to_pop_map']
            song_to_fam_map = saved_mappings['song_to_fam_map']
            song_to_genres_map = saved_mappings['song_to_genres_map']
            song_to_language_map = saved_mappings['song_to_language_map']
            related_genres_map = saved_mappings['related_genres_map']
            song_to_artist_map = saved_mappings['song_to_artist_map']
            logger.info("Mappings loaded successfully from cache.")
        except Exception as e:
            logger.warning(f"Error loading saved mappings: {e}. Recomputing mappings.")
    
    if not song_to_tier_map:
        logger.info("Creating global mappings based on training data...")
        song_to_tier_map = create_song_to_tier_mapping(full_song_metadata)
        song_to_pop_map, song_to_fam_map = create_artist_metric_mappings(full_song_metadata, artist_identification_raw)
        song_to_genres_map = create_genre_mapping(full_song_metadata)
        song_to_language_map = create_language_mapping(full_song_metadata)
        related_genres_map = compute_related_genres(song_to_genres_map)
        song_to_artist_map = dict(zip(full_song_metadata['song_id'], full_song_metadata.get('artist_name', pd.Series(index=full_song_metadata['song_id'], dtype=str))))
        logger.info("Global mappings created.")

        if config.get('saved_data_dir'):
            os.makedirs(config['saved_data_dir'], exist_ok=True)
            mappings = {
                'song_to_tier_map': song_to_tier_map,
                'song_to_pop_map': song_to_pop_map,
                'song_to_fam_map': song_to_fam_map,
                'song_to_genres_map': song_to_genres_map,
                'song_to_language_map': song_to_language_map,
                'related_genres_map': related_genres_map,
                'song_to_artist_map': song_to_artist_map
            }
            try:
                with open(mappings_path, 'wb') as f:
                    pickle.dump(mappings, f)
                logger.info(f"Saved mappings to {mappings_path}")
            except Exception as e:
                logger.error(f"Error saving mappings to cache: {e}")

    # Initialize ALL recommender instances
    content_recommender = ContentBasedRecommender(
        tier_weights=config['tier_weights'],
        user_profiles={},
        user_extended_genres_cache={},
        related_genres_map=related_genres_map,
        song_to_tier=song_to_tier_map,
        song_to_pop=song_to_pop_map,
        song_to_fam=song_to_fam_map,
        song_to_genres=song_to_genres_map,
        song_to_language=song_to_language_map,
        song_to_artist=song_to_artist_map
    )
    cf_recommender = CollaborativeFilteringRecommender(
        tier_weights=config['tier_weights'],
        song_to_tier=song_to_tier_map,
        song_to_pop=song_to_pop_map,
        song_to_fam=song_to_fam_map,
        song_to_genres=song_to_genres_map,
        song_to_language=song_to_language_map,
        user_profiles={},
        user_extended_genres_cache={},
        related_genres_map=related_genres_map
    )
    mf_recommender = MatrixFactorizationRecommender(
        tier_weights=config['tier_weights'],
        song_to_tier=song_to_tier_map,
        song_to_pop=song_to_pop_map,
        song_to_fam=song_to_fam_map,
        song_to_genres=song_to_genres_map,
        song_to_language=song_to_language_map,
        user_profiles={},
        user_extended_genres_cache={},
        related_genres_map=related_genres_map,
        song_to_artist=song_to_artist_map
    )
    
    hybrid_recommender = HybridRecommender(
        content_recommender=content_recommender,
        mf_recommender=mf_recommender,
        tier_weights=config['tier_weights'],
        content_weight=config.get('hybrid_content_weight', 0.5),
        mf_weight=config.get('hybrid_mf_weight', 0.5),
        song_to_artist=song_to_artist_map
    )

    # NEW LOGIC START: Always add all recommender objects to shared_recommenders
    # This ensures they are available in shared_data for component loading,
    # even if they are not explicitly run for optimization/evaluation in this specific run.
    shared_recommenders = {
        'content_recommender': content_recommender,
        'cf_recommender': cf_recommender,
        'mf_recommender': mf_recommender,
        'hybrid_recommender': hybrid_recommender
    }
    # NEW LOGIC END

    # Conditionally add recommenders to the system based on config['recommender_types_to_run']
    # This part determines which recommenders are "active" for the system's overall management.
    for name in config.get('recommender_types_to_run', []):
        if name == 'ContentBased':
            system.add_recommender(shared_recommenders['content_recommender'])
        elif name == 'CollaborativeFiltering':
            system.add_recommender(shared_recommenders['cf_recommender'])
        elif name == 'MatrixFactorization':
            system.add_recommender(shared_recommenders['mf_recommender'])
        elif name == 'HybridContentMF':
            system.add_recommender(shared_recommenders['hybrid_recommender'])
        else:
            logger.warning(f"Recommender type '{name}' in recommender_types_to_run is not recognized or initialized.")


    os.makedirs(config['cache_dir'], exist_ok=True)
    logger.info(f"Ensured cache directory exists: {config['cache_dir']}")

    recommender_types_to_train_initially = config.get('recommender_types_to_run', [])
    for name in recommender_types_to_train_initially:
        recommender_obj = None
        if name == 'ContentBased':
            recommender_obj = shared_recommenders['content_recommender']
        elif name == 'CollaborativeFiltering':
            recommender_obj = shared_recommenders['cf_recommender']
        elif name == 'MatrixFactorization':
            recommender_obj = shared_recommenders['mf_recommender']
        elif name == 'HybridContentMF':
            recommender_obj = shared_recommenders['hybrid_recommender']
        
        if recommender_obj:
            recommender_obj.train(
                user_interactions=train_interactions,
                song_features=full_song_features,
                song_metadata=full_song_metadata,
                artist_identification=artist_identification_raw,
                testing_mode=config['verbose'],
                cache_dir=config['cache_dir']
            )
            logger.debug(f"After {name} train, user_profiles has {len(getattr(recommender_obj, 'user_profiles', {}))} entries.")
        else:
            logger.warning(f"Recommender '{name}' not found in shared_recommenders for initial explicit training. Skipping.")
    logger.info("Recommenders initialized and initially trained with training data.")

    return {
        'system': system,
        'content_recommender': shared_recommenders['content_recommender'],
        'cf_recommender': shared_recommenders['cf_recommender'],
        'mf_recommender': shared_recommenders['mf_recommender'],
        'hybrid_recommender': shared_recommenders['hybrid_recommender'],
        'train_interactions': train_interactions,
        'test_interactions': test_interactions,
        'song_metadata': full_song_metadata,
        'song_features': full_song_features,
        'song_to_tier_map': song_to_tier_map,
        'song_to_pop_map': song_to_pop_map,
        'song_to_fam_map': song_to_fam_map,
        'song_to_genres_map': song_to_genres_map,
        'song_to_language_map': song_to_language_map,
        'related_genres_map': related_genres_map,
        'song_to_artist_map': song_to_artist_map
    }

def generate_recommendations_for_user(user_id, k, config, shared_data, recommender_key):
    """Generate recommendations for a single user."""
    try:
        recommender = shared_data.get(recommender_key)
        if recommender is None:
            logger.error(f"Recommender '{recommender_key}' not found in shared_data. Cannot generate recommendations.")
            return None

        recs = recommender.recommend(
            user_id=user_id,
            n=k,
            exclude_listened=False,
            include_metadata=True,
            user_interactions=shared_data['train_interactions'],
            testing_mode=config['verbose']
        )
        return recs if not recs.empty else None
    except Exception as e:
        logger.error(f"Error generating recommendations for user {user_id} with {recommender_key}: {str(e)}")
        return None

def sequential_recommendations(config, eval_users, k, shared_data, recommender_key):
    """Generate recommendations sequentially."""
    recommender_name_map = {
        'content_recommender': 'ContentBased',
        'cf_recommender': 'CollaborativeFiltering',
        'mf_recommender': 'MatrixFactorization',
        'hybrid_recommender': 'HybridContentMF'
    }
    recommender_name = recommender_name_map.get(recommender_key, recommender_key)
    logger.info(f"Starting sequential recommendation generation for {recommender_name} with k={k} for {len(eval_users)} users.")

    recs_list = []
    for user_id in eval_users:
        recs = generate_recommendations_for_user(user_id, k, config, shared_data, recommender_key)
        if recs is not None:
            recs_list.append(recs)

    valid_recs = [recs for recs in recs_list if recs is not None and not recs.empty]
    if not valid_recs:
        logger.error(f"No valid recommendations generated for {recommender_name} with k={k}. Metrics will be 0.0 for this k.")
        return pd.DataFrame()

    total_recs_generated = sum(len(df) for df in valid_recs)
    logger.info(f"Successfully generated {total_recs_generated} recommendations for {recommender_name} with k={k} from {len(valid_recs)} users.")
    return pd.concat(valid_recs, ignore_index=True)

def _load_and_apply_component_tier_weights(recommender_obj, component_name, k_val, config, shared_data):
    """Helper function to load and apply optimized tier weights for a component recommender."""
    # Map the component_name (e.g., 'ContentBased') to its key in shared_data (e.g., 'content_recommender')
    component_key_map = {
        'ContentBased': 'content_recommender',
        'MatrixFactorization': 'mf_recommender',
        'CollaborativeFiltering': 'cf_recommender' # Although not directly used by Hybrid, keep consistent
    }
    component_recommender_key = component_key_map.get(component_name)

    if component_recommender_key is None:
        logger.warning(f"Invalid component name '{component_name}' provided. Cannot load/apply optimized tier weights.")
        return

    component_recommender = shared_data.get(component_recommender_key)
    
    if component_recommender is None:
        logger.warning(f"Component recommender '{component_name}' (key: '{component_recommender_key}') not found in shared_data. Cannot load/apply optimized tier weights.")
        return

    db_file = os.path.join(config['optuna_db_path'], f"optuna_study_{component_name}_k{k_val}.db")
    
    if config.get('load_optimized_weights', False) and os.path.exists(db_file):
        component_best_weights, _ = load_optuna_weights(db_file, f"study_{component_name}_k{k_val}")
        if component_best_weights:
            component_recommender.tier_weights = component_best_weights
            logger.info(f"Applied optimized tier weights to {component_name} component of {recommender_obj.name} (k={k_val}): {component_best_weights}")
        else:
            logger.warning(f"Failed to load optimized tier weights for {component_name} (k={k_val}). Using default tier_weights for component.")
            component_recommender.tier_weights = config['tier_weights']
    else:
        logger.info(f"No optimized tier weights to load for {component_name} (k={k_val}). Using default tier_weights for component.")
        component_recommender.tier_weights = config['tier_weights']


def perform_evaluation_run(config: dict, shared_data, recommender_key):
    """Performs a single evaluation run with the given config and shared data."""
    recommender_name_map = {
        'content_recommender': 'ContentBased',
        'cf_recommender': 'CollaborativeFiltering',
        'mf_recommender': 'MatrixFactorization',
        'hybrid_recommender': 'HybridContentMF'
    }
    recommender_name = recommender_name_map.get(recommender_key, recommender_key)
    metrics = {}
    exposure_by_k = {}
    tier_diversity_by_k = {}
    gini_by_k = {}

    recommender = shared_data.get(recommender_key)
    if recommender is None:
        logger.error(f"Recommender '{recommender_key}' not found in shared_data for evaluation. Skipping.")
        return {}, {}

    test_interactions = shared_data['test_interactions']
    eval_users = config['eval_users']

    if not eval_users:
        logger.error(f"No valid evaluation users found for {recommender_name}. Cannot run recommendation or evaluation.")
        return {}, {}

    logger.info(f"Number of evaluation users selected for {recommender_name}: {len(eval_users)}")

    for k in config['k_values']:
        if recommender_key == 'hybrid_recommender':
            logger.info(f"Applying optimized tier weights to Hybrid components for k={k}...")
            _load_and_apply_component_tier_weights(recommender, 'ContentBased', k, config, shared_data)
            _load_and_apply_component_tier_weights(recommender, 'MatrixFactorization', k, config, shared_data)

        eval_recs = sequential_recommendations(config, eval_users, k, shared_data, recommender_key)

        if eval_recs.empty:
            logger.error(f"SKIPPING EVALUATION: No recommendations generated for {recommender_name} with k={k}. Metrics will be 0.0 for this k.")
            for metric_name in ['Precision', 'Genre Precision', 'Language Precision', 'Recall', 'NDCG', 'Hit Rate', 'Emerging Artist Hit Rate', 'Coverage (%)', 'Diversity', 'Novelty', 'Emerging Artist Exposure Index']:
                if metric_name not in metrics:
                    metrics[metric_name] = {}
                metrics[metric_name][k] = 0.0
            tier_diversity_by_k[k] = 0.0
            gini_by_k[k] = 0.0
            continue

        shared_data['system'].track_exposure(eval_recs, recommender.song_to_artist)
        exposure_by_k[k] = shared_data['system'].artist_exposure.copy()

        if recommender_key == 'content_recommender':
            evaluator = RecommendationEvaluator(
                item_features=recommender.item_features,
                item_ids=recommender.item_ids,
                song_metadata=recommender.song_metadata,
                k_values=[k]
            )
        elif recommender_key in ['cf_recommender', 'mf_recommender', 'hybrid_recommender']:
            evaluator = CFRecommendationEvaluator(
                song_features=shared_data['song_features'],
                item_ids=list(recommender.song_id_to_index.keys()) if hasattr(recommender, 'song_id_to_index') else list(shared_data['song_metadata']['song_id'].unique()),
                song_metadata=recommender.song_metadata,
                k_values=[k]
            )
        else:
            raise ValueError(f"Unknown recommender key for evaluation: {recommender_key}")
        
        current_k_metrics = evaluator.evaluate(recommendations=eval_recs, test_interactions=test_interactions)
        for metric_name, values in current_k_metrics.items():
            if metric_name not in metrics:
                metrics[metric_name] = {}
            metrics[metric_name][k] = values.get(k, 0.0)

        exposure_result = shared_data['system'].analyze_exposure_distribution(
            song_metadata=recommender.song_metadata
        )
        tier_diversity_by_k[k] = exposure_result.get('tier_diversity', 0.0)
        gini_by_k[k] = exposure_result.get('gini_coefficient', 0.0)

    loss = {}
    if any(k in m for m in metrics.values() for k in config['k_values']):
        loss_calculator = config['loss_calculators'][recommender_name]
        loss_results_dict, ls_results_dict, as_results_dict = loss_calculator.compute_objective_loss(
            metrics=metrics,
            k_values=config['k_values'],
            tier_diversity_by_k=tier_diversity_by_k,
            gini_by_k=gini_by_k
        )
        loss = loss_results_dict 
    else:
        logger.error(f"No valid metrics found to compute objective loss for {recommender_name}. Returning empty loss.")

    return metrics, loss

def execute_training(config, shared_data):
    """Executes the training phase for specified recommenders."""
    logger.info("\n--- Starting Training Phase ---")
    recommender_names = config.get('recommender_types_to_run', [])
    if not recommender_names:
        logger.info("No recommenders specified for training. Skipping training phase.")
        return
    logger.info("Recommenders were trained during resource initialization using the training data.")
    logger.info("--- Training Phase Complete ---")

def execute_optimization(config, shared_data):
    """Executes the optimization phase for recommenders' weights."""
    logger.info("\n--- Starting Optimization Process ---")
    
    if not config.get('run_optimization', False):
        logger.info("Optimization is not enabled in config. Skipping optimization phase.")
        return {}

    os.makedirs(config['tensorboard_log_dir'], exist_ok=True)
    logger.info(f"Ensured TensorBoard log directory exists: {config['tensorboard_log_dir']}")

    optimized_weights_per_recommender = {}
    tiers_for_optimization = [
        'emerging_new', 'emerging_trending', 'rising_established',
        'mid_tier', 'established', 'established_trending', 'established_legacy'
    ]
    
    recommender_keys_to_optimize = []
    if 'ContentBased' in config.get('recommender_types_to_run', []):
        recommender_keys_to_optimize.append('content_recommender')
    if 'CollaborativeFiltering' in config.get('recommender_types_to_run', []):
        recommender_keys_to_optimize.append('cf_recommender')
    if 'MatrixFactorization' in config.get('recommender_types_to_run', []):
        recommender_keys_to_optimize.append('mf_recommender')
    if 'HybridContentMF' in config.get('recommender_types_to_run', []):
        recommender_keys_to_optimize.append('hybrid_recommender')


    if not recommender_keys_to_optimize:
        logger.info("No optimizable recommenders specified. Skipping optimization.")
        return {}

    for recommender_key in recommender_keys_to_optimize:
        recommender = shared_data.get(recommender_key)
        if recommender is None:
            logger.warning(f"Recommender '{recommender_key}' not found in shared_data. Skipping optimization for this recommender.")
            continue

        recommender_name_map = {
            'content_recommender': 'ContentBased',
            'cf_recommender': 'CollaborativeFiltering',
            'mf_recommender': 'MatrixFactorization',
            'hybrid_recommender': 'HybridContentMF'
        }
        recommender_name = recommender_name_map.get(recommender_key, recommender_key)

        for k_val in config['k_values']:
            logger.info(f"Running optimization for {recommender_name} at k={k_val}")
            shared_data['system'].artist_exposure = {}
            
            if recommender_key == 'hybrid_recommender':
                logger.info(f"Applying optimized tier weights to Hybrid components for k={k_val} during optimization...")
                _load_and_apply_component_tier_weights(recommender, 'ContentBased', k_val, config, shared_data)
                _load_and_apply_component_tier_weights(recommender, 'MatrixFactorization', k_val, config, shared_data)

            if recommender_key == 'content_recommender':
                evaluator = RecommendationEvaluator(
                    item_features=recommender.item_features,
                    item_ids=recommender.item_ids,
                    song_metadata=recommender.song_metadata,
                    k_values=config['k_values']
                )
            elif recommender_key in ['cf_recommender', 'mf_recommender', 'hybrid_recommender']:
                evaluator = CFRecommendationEvaluator(
                    song_features=shared_data['song_features'],
                    item_ids=list(recommender.song_id_to_index.keys()) if hasattr(recommender, 'song_id_to_index') else list(shared_data['song_metadata']['song_id'].unique()),
                    song_metadata=recommender.song_metadata,
                    k_values=config['k_values']
                )

            is_hybrid_opt = (recommender_key == 'hybrid_recommender')
            best_weights, best_loss, best_metrics_from_opt = optimize_tier_weights(
                config=config,
                system=shared_data['system'],
                recommender=recommender,
                evaluator=evaluator,
                train_interactions=shared_data['train_interactions'],
                test_interactions=shared_data['test_interactions'],
                eval_users=config['eval_users'],
                tiers=tiers_for_optimization,
                recommender_name=recommender_name,
                weight_options=config['weight_options'],
                optimization_method=config['optimization_method'],
                k=k_val,
                n_trials=config['n_trials'],
                is_hybrid_optimization=is_hybrid_opt
            )
            if recommender_name not in optimized_weights_per_recommender:
                optimized_weights_per_recommender[recommender_name] = {}
            optimized_weights_per_recommender[recommender_name][k_val] = (best_weights, best_loss, best_metrics_from_opt)
            print(f"\nOptimization for {recommender_name} at k={k_val} complete. Best loss: {best_loss:.4f}")
            print(f"Best weights: {best_weights}")
    logger.info("--- Optimization Process Complete ---")
    return optimized_weights_per_recommender

def execute_evaluation(config, shared_data, best_weights_per_recommender):
    """Executes the evaluation phase for specified recommenders."""
    logger.info("\n--- Starting Evaluation Phase ---")
    
    metrics_path = os.path.join(config['saved_data_dir'], f"metrics_u{config['max_users']}.pkl") if config.get('saved_data_dir') else None

    if config['load_saved_data'] and metrics_path and os.path.exists(metrics_path):
        logger.info(f"Attempting to load saved metrics for {config['max_users']} users from {metrics_path}. Will re-compute for display.")
        with open(metrics_path, 'rb') as f:
            saved_metrics = pickle.load(f)

    final_metrics_per_recommender = {}
    final_loss_per_recommender = {}

    recommender_keys_to_evaluate = []
    recommender_names_from_config = config.get('recommender_types_to_run', [])
    if 'ContentBased' in recommender_names_from_config:
        recommender_keys_to_evaluate.append('content_recommender')
    if 'CollaborativeFiltering' in recommender_names_from_config:
        recommender_keys_to_evaluate.append('cf_recommender')
    if 'MatrixFactorization' in recommender_names_from_config:
        recommender_keys_to_evaluate.append('mf_recommender')
    if 'HybridContentMF' in recommender_names_from_config:
        recommender_keys_to_evaluate.append('hybrid_recommender')

    if not recommender_keys_to_evaluate:
        logger.info("No recommenders specified for evaluation. Skipping evaluation phase.")
        return

    if not config['eval_users']:
        logger.warning("Evaluation skipped: No valid evaluation users available.")
        return

    for k_val in config['k_values']:
        print(f"\n--- Running evaluation for k={k_val} ---")
        current_config_for_k = config.copy()
        current_config_for_k['k_values'] = [k_val]

        for recommender_key in recommender_keys_to_evaluate:
            recommender = shared_data.get(recommender_key)
            if recommender is None:
                logger.warning(f"Recommender '{recommender_key}' not found in shared_data for evaluation. Skipping.")
                continue

            recommender_name_map = {
                'content_recommender': 'ContentBased',
                'cf_recommender': 'CollaborativeFiltering',
                'mf_recommender': 'MatrixFactorization',
                'hybrid_recommender': 'HybridContentMF'
            }
            recommender_name = recommender_name_map.get(recommender_key, recommender_key)

            if recommender_name in best_weights_per_recommender and k_val in best_weights_per_recommender[recommender_name]:
                weights_for_this_recommender_k = best_weights_per_recommender[recommender_name][k_val][0]
                if recommender_name == 'HybridContentMF':
                    recommender.content_weight = weights_for_this_recommender_k.get('content_weight', 0.5)
                    recommender.mf_weight = weights_for_this_recommender_k.get('mf_weight', 0.5)
                    logger.info(f"Using hybrid weights for {recommender_name} at k={k_val}: Content={recommender.content_weight:.2f}, MF={recommender.mf_weight:.2f}")
                    
                    logger.info(f"Applying optimized tier weights to Hybrid components for k={k_val} during final evaluation...")
                    _load_and_apply_component_tier_weights(recommender, 'ContentBased', k_val, config, shared_data)
                    _load_and_apply_component_tier_weights(recommender, 'MatrixFactorization', k_val, config, shared_data)

                else:
                    recommender.tier_weights = weights_for_this_recommender_k
                    logger.info(f"Using tier weights for {recommender_name} at k={k_val}: {recommender.tier_weights}")
            else:
                logger.warning(f"No optimized weights found for {recommender_name} at k={k_val}. Using default config weights.")
                if recommender_name == 'HybridContentMF':
                    recommender.content_weight = config.get('hybrid_content_weight', 0.5)
                    recommender.mf_weight = config.get('hybrid_mf_weight', 0.5)
                    logger.info(f"Using default tier weights for Hybrid components for k={k_val} during final evaluation (no optimized found)...")
                    _load_and_apply_component_tier_weights(recommender, 'ContentBased', k_val, config, shared_data)
                    _load_and_apply_component_tier_weights(recommender, 'MatrixFactorization', k_val, config, shared_data)
                else:
                    recommender.tier_weights = config['tier_weights']

            shared_data['system'].artist_exposure = {}
            metrics_from_run, loss_from_run = perform_evaluation_run(current_config_for_k, shared_data, recommender_key)

            if recommender_name not in final_metrics_per_recommender:
                final_metrics_per_recommender[recommender_name] = {}
            if recommender_name not in final_loss_per_recommender:
                final_loss_per_recommender[recommender_name] = {}

            if metrics_from_run:
                for metric_name, val_dict in metrics_from_run.items():
                    if metric_name not in final_metrics_per_recommender[recommender_name]:
                        final_metrics_per_recommender[recommender_name][metric_name] = {}
                    final_metrics_per_recommender[recommender_name][metric_name][k_val] = val_dict.get(k_val, 0.0)
                
                final_loss_per_recommender[recommender_name][k_val] = loss_from_run.get(k_val, 0.0)
            else:
                logger.error(f"Failed to get valid metrics for {recommender_name} with k={k_val}. Metrics will be 0.0.")
                for metric_name in ['Precision', 'Genre Precision', 'Language Precision', 'Recall', 'NDCG', 'Hit Rate', 'Emerging Artist Hit Rate', 'Coverage (%)', 'Diversity', 'Novelty', 'Emerging Artist Exposure Index']:
                    if metric_name not in final_metrics_per_recommender[recommender_name]:
                        final_metrics_per_recommender[recommender_name][metric_name] = {}
                    final_metrics_per_recommender[recommender_name][metric_name][k_val] = 0.0
                final_loss_per_recommender[recommender_name][k_val] = 0.0

    if config.get('saved_data_dir'):
        os.makedirs(config['saved_data_dir'], exist_ok=True)
        metrics_data = {
            'final_metrics_per_recommender': final_metrics_per_recommender,
            'final_loss_per_recommender': final_loss_per_recommender
        }
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics_data, f)
        logger.info(f"Saved metrics to {metrics_path}")

    logger.info("\n--- Final Metrics with Weights (Across all k values) ---")
    sorted_k_values = sorted(config['k_values'])
    for recommender_name in config.get('recommender_types_to_run', []):
        if recommender_name in final_metrics_per_recommender:
            metrics_to_print = final_metrics_per_recommender[recommender_name]
            loss_to_print = final_loss_per_recommender[recommender_name]
            print(f"\nResults for {recommender_name.upper()}")
            format_metrics(metrics_to_print, sorted_k_values, loss_to_print)
        else:
            logger.info(f"Skipping final metrics display for {recommender_name} as it was not evaluated.")

    logger.info("--- Evaluation Phase Complete ---")

def execute_recommendation_comparison(config, shared_data, best_weights_per_recommender):
    """Performs a recommendation comparison for a random user."""
    logger.info("\n--- Starting Recommendation Comparison Phase ---")
    
    recommender_names_to_compare = config.get('recommender_types_to_run', [])
    if not recommender_names_to_compare:
        logger.info("No recommenders specified for comparison. Skipping comparison phase.")
        return

    available_users_with_history = shared_data['train_interactions']['user_id'].unique()
    if len(available_users_with_history) == 0:
        logger.error("No users with interaction history available in training data to select for comparison.")
        return

    np.random.seed(int(time.time()))
    sample_user_id = np.random.choice(available_users_with_history)
    logger.info(f"Selected random user ID for comparison: {sample_user_id}")

    user_history = shared_data['train_interactions'][shared_data['train_interactions']['user_id'] == sample_user_id]
    
    if user_history.empty:
        logger.error(f"Critical Error: Selected user {sample_user_id} has no interaction history in train_interactions.")
        return

    print("\n" + "="*80)
    print(f"USER LISTENING HISTORY FOR USER {sample_user_id}")
    print("="*80)
    history_profile = analyze_user_listening_history(
        user_interactions=user_history,
        song_metadata=shared_data['song_metadata']
    )
    print(history_profile['formatted'])

    print("\n" + "="*80)
    print(f"USER LISTENING HISTORY SONGS FOR USER {sample_user_id}")
    print("="*80)

    genre_column = 'top_genre'
    language_column = 'language'

    if genre_column not in user_history.columns:
        logger.error(f"Genre column '{genre_column}' not found in user_history. Available columns: {list(user_history.columns)}")
        user_history[genre_column] = 'unknown_genre'
    if language_column not in user_history.columns:
        logger.error(f"Language column '{language_column}' not found in user_history. Available columns: {list(user_history.columns)}")
        user_history[language_column] = 'unknown_language'

    history_with_metadata = user_history.merge(
        shared_data['song_metadata'][['song_id', 'title', 'artist_name']],
        on='song_id',
        how='left'
    )

    history_with_metadata[genre_column] = history_with_metadata[genre_column].fillna('unknown_genre')
    history_with_metadata[language_column] = history_with_metadata[language_column].fillna('unknown_language')
    history_with_metadata['title'] = history_with_metadata['title'].fillna('unknown_title')
    history_with_metadata['artist_name'] = history_with_metadata['artist_name'].fillna('unknown_artist')

    if history_with_metadata.empty or history_with_metadata[['title', 'artist_name']].isna().all().all():
        logger.warning(f"No metadata available for user {sample_user_id}'s listening history songs")
        print("No song metadata available to display.")
    else:
        history_with_metadata = history_with_metadata.drop_duplicates(subset=['song_id'])
        print_recommendation_details(
            recommendations=history_with_metadata,
            header=f"USER LISTENING HISTORY SONGS",
            max_items=20,
            columns_to_display=['song_id', 'title', 'artist_name', genre_column, language_column]
        )
        logger.info(f"Displayed history for user {sample_user_id} with {len(history_with_metadata)} unique songs")
    print("="*80)

    valid_songs = user_history[user_history['song_id'].isin(shared_data['song_metadata']['song_id'])]
    if valid_songs.empty:
        logger.error(f"No valid seed songs found from user {sample_user_id}'s history that are also in song_metadata.")
        return
    sample_song_id = valid_songs['song_id'].iloc[0]
    logger.info(f"Selected seed song ID: {sample_song_id}")

    shared_data['system'].artist_exposure = {}

    max_k_for_comparison = max(config['k_values'])
    recommender_key_map = {
        'ContentBased': 'content_recommender',
        'CollaborativeFiltering': 'cf_recommender',
        'MatrixFactorization': 'mf_recommender',
        'HybridContentMF': 'hybrid_recommender'
    }

    for recommender_name in recommender_names_to_compare:
        recommender_data_key = recommender_key_map.get(recommender_name)
        if not recommender_data_key:
            logger.warning(f"Recommender name '{recommender_name}' not found in key map. Skipping comparison.")
            continue

        recommender_obj = shared_data.get(recommender_data_key)
        if recommender_obj is None:
            logger.warning(f"Recommender '{recommender_data_key}' not found in shared_data for comparison. Skipping.")
            continue

        if recommender_name in best_weights_per_recommender and max_k_for_comparison in best_weights_per_recommender[recommender_name]:
            weights_for_this_recommender = best_weights_per_recommender[recommender_name][max_k_for_comparison][0]
            if recommender_name == 'HybridContentMF':
                recommender_obj.content_weight = weights_for_this_recommender.get('content_weight', 0.5)
                recommender_obj.mf_weight = weights_for_this_recommender.get('mf_weight', 0.5)
                logger.info(f"Using hybrid weights for comparison for {recommender_name}: Content={recommender_obj.content_weight:.2f}, MF={recommender_obj.mf_weight:.2f}")

                logger.info(f"Applying optimized tier weights to Hybrid components for comparison...")
                _load_and_apply_component_tier_weights(recommender_obj, 'ContentBased', max_k_for_comparison, config, shared_data)
                _load_and_apply_component_tier_weights(recommender_obj, 'MatrixFactorization', max_k_for_comparison, config, shared_data)

            else:
                recommender_obj.tier_weights = weights_for_this_recommender
                logger.info(f"Using weights for comparison for {recommender_name}: {recommender_obj.tier_weights}")
        else:
            logger.warning(f"No optimized weights found for {recommender_name} at k={max_k_for_comparison}. Using default config weights.")
            if recommender_name == 'HybridContentMF':
                recommender_obj.content_weight = config.get('hybrid_content_weight', 0.5)
                recommender_obj.mf_weight = config.get('hybrid_mf_weight', 0.5)
                logger.info(f"Using default tier weights for Hybrid components for comparison (no optimized found)...")
                _load_and_apply_component_tier_weights(recommender_obj, 'ContentBased', max_k_for_comparison, config, shared_data)
                _load_and_apply_component_tier_weights(recommender_obj, 'MatrixFactorization', max_k_for_comparison, config, shared_data)
            else:
                recommender_obj.tier_weights = config['tier_weights']


        print("\n" + "="*80)
        print(f"RECOMMENDATION APPROACH COMPARISON FOR {recommender_name.upper()}")
        print("="*80)
        try:
            user_item_recs, item_item_recs = compare_recommendation_approaches(
                system=shared_data['system'],
                user_id=sample_user_id,
                seed_item_id=sample_song_id,
                recommender=recommender_obj,
                n=config['n_recs'],
                testing_mode=config['verbose']
            )
            if not user_item_recs.empty:
                print_recommendation_details(
                    recommendations=user_item_recs,
                    header="USER-BASED RECOMMENDATIONS",
                    max_items=config['n_recs'],
                    columns_to_display=['song_id', 'title', 'artist_name', genre_column, language_column]
                )
            else:
                logger.warning(f"No user-based recommendations generated for {recommender_name}.")
                print("\nNo user-based recommendations to display.")
            if not item_item_recs.empty:
                print_recommendation_details(
                    recommendations=item_item_recs,
                    header="ITEM-BASED RECOMMENDATIONS",
                    max_items=config['n_recs'],
                    columns_to_display=['song_id', 'title', 'artist_name', genre_column, language_column]
                )
            else:
                logger.warning(f"No item-based recommendations generated for {recommender_name}.")
                print("\nNo item-based recommendations to display.")
        except Exception as e:
            logger.error(f"Error comparing recommendations for {recommender_name}: {str(e)}")
            print(f"Unable to compare recommendations for {recommender_name}: {str(e)}")
        print("\n" + "="*80)
        print(f"COMPARISON COMPLETE FOR {recommender_name.upper()}")
        print("="*80)

def main_run(config):
    """Main function to run the recommendation system based on config."""
    logger.info("--- Initializing resources ---")

    overall_log_dir = os.path.join(config['tensorboard_log_dir'], 'overall_run')
    os.makedirs(overall_log_dir, exist_ok=True)
    overall_writer = tf.summary.create_file_writer(overall_log_dir)
    
    with overall_writer.as_default():
        tf.summary.text("Run Status", "Started main recommendation system execution", step=0)
        tf.summary.scalar("Overall_Start_Timestamp", time.time(), step=0)

    shared_data = initialize_shared_resources(config)
    
    initial_eval_users_from_test = shared_data['test_interactions']['user_id'].unique()
    user_counts = shared_data['train_interactions']['user_id'].value_counts()
    config['eval_users'] = user_counts[user_counts.index.isin(initial_eval_users_from_test)].index.tolist()

    if not config['eval_users']:
        logger.critical("No initial evaluation users found for evaluation or comparison. These phases will be skipped.")
        with overall_writer.as_default():
            tf.summary.text("Run Status", "Skipped evaluation and comparison due to no valid users", step=1)
            tf.summary.scalar("Overall_End_Timestamp", time.time(), step=1)
            tf.summary.scalar("Overall_Run_Success", 0.0, step=1)
        overall_writer.flush()
        overall_writer.close()
        return

    best_weights_per_recommender = {}

    if config.get('run_optimization', False):
        optimized_results = execute_optimization(config, shared_data)
        if optimized_results:
            for recommender_name, k_results in optimized_results.items():
                if recommender_name not in best_weights_per_recommender:
                    best_weights_per_recommender[recommender_name] = {}
                for k_val, (weights, loss, metrics) in k_results.items():
                    best_weights_per_recommender[recommender_name][k_val] = (weights, loss, metrics)
        else:
            logger.warning("Optimization phase did not return results.")
    
    elif config.get('load_optimized_weights', False):
        logger.info("--- Loading Optimized Weights for Evaluation ---")
        recommender_types_to_load = config.get('recommender_types_to_run', [])

        for recommender_name in recommender_types_to_load:
            if recommender_name not in best_weights_per_recommender:
                best_weights_per_recommender[recommender_name] = {}
            for k_val in config['k_values']:
                db_file = os.path.join(config['optuna_db_path'], f"optuna_study_{recommender_name}_k{k_val}.db") 
                if os.path.exists(db_file):
                    best_weights, best_loss = load_optuna_weights(db_file, f"study_{recommender_name}_k{k_val}")
                    if best_weights:
                        best_metrics_from_opt = {} 
                        best_weights_per_recommender[recommender_name][k_val] = (best_weights, best_loss, best_metrics_from_opt)
                        logger.info(f"Loaded optimized weights for {recommender_name} at k={k_val}: {best_weights}")
                    else:
                        logger.warning(f"Failed to load optimized weights for {recommender_name} at k={k_val}. Using default base_config weights.")
                        best_weights_per_recommender[recommender_name][k_val] = (config['tier_weights'], float('inf'), {})
                else:
                    logger.warning(f"Optuna DB file for {recommender_name} at k={k_val} not found. Using default base_config weights.")
                    best_weights_per_recommender[recommender_name][k_val] = (config['tier_weights'], float('inf'), {})

    else:
        logger.info("--- Using Default Tier Weights for All Recommenders and K-values ---")
        for recommender_name in config.get('recommender_types_to_run', []):
            if recommender_name not in best_weights_per_recommender:
                best_weights_per_recommender[recommender_name] = {}
            for k_val in config['k_values']:
                if recommender_name == 'HybridContentMF':
                    best_weights_per_recommender[recommender_name][k_val] = (
                        {'content_weight': config.get('hybrid_content_weight', 0.5),
                         'mf_weight': config.get('hybrid_mf_weight', 0.5)},
                        float('inf'), {}
                    )
                else:
                    best_weights_per_recommender[recommender_name][k_val] = (config['tier_weights'], float('inf'), {})


    execute_training(config, shared_data)
    execute_evaluation(config, shared_data, best_weights_per_recommender)
    execute_recommendation_comparison(config, shared_data, best_weights_per_recommender)

    logger.info("\nAll configured processes complete.")
    
    with overall_writer.as_default():
        tf.summary.text("Run Status", "Completed main recommendation system execution successfully", step=1)
        tf.summary.scalar("Overall_End_Timestamp", time.time(), step=1)
        tf.summary.scalar("Overall_Run_Success", 1.0, step=1)
    overall_writer.flush()
    overall_writer.close()

if __name__ == "__main__":
    base_config = {
        # Paths relative to repo root for portability
        'data_path': os.path.join(project_root, 'System', 'data'),
        'optuna_db_path': os.path.join(project_root, 'System', 'models'),
        'saved_data_dir': os.path.join(project_root, 'System', 'SavedData2'),
        'load_saved_data': True,
        'run_optimization': False,
        'load_optimized_weights': True,
        'max_users': 400,
        'k_values': [5],
        'n_recs': 5,
        'min_interactions': 6,
        'verbose': False,
        'tier_weights': {
            'emerging_new': 0.15789473684210525,
            'emerging_trending': 0.2105263157894737,
            'rising_established': 0.15789473684210525,
            'mid_tier': 0.10526315789473685,
            'established': 0.15789473684210525,
            'established_trending': 0.10526315789473685,
            'established_legacy': 0.10526315789473685
        },
        'hybrid_content_weight': 0.38,
        'hybrid_mf_weight': 0.62,
        'optimization_method': 'bayesian',
        'weight_options': [0.1, 0.15, 0.2],
        'n_trials': 200,
        'loss_calculators': {
            'ContentBased': ObjectiveLossCalculator(recommender_name='ContentBased'),
            'CollaborativeFiltering': ObjectiveLossCalculator(recommender_name='CollaborativeFiltering'),
            'MatrixFactorization': ObjectiveLossCalculator(recommender_name='MatrixFactorization'),
            'HybridContentMF': ObjectiveLossCalculator(recommender_name='HybridContentMF')
        },
        'recommender_types_to_run': ['CollaborativeFiltering'],
        'cache_dir': os.path.join(project_root, 'System', 'SavedData', 'cache'),
        'tensorboard_log_dir': os.path.join(project_root, 'System', 'TensorBoardLogs'),
    }
    main_run(base_config)
