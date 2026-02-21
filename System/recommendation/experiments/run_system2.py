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

    if genre_column:
        user_genres = filtered_interactions.groupby('user_id')[genre_column].nunique()
        valid_users_genre_filtered = user_genres[user_genres <= 2].index
        valid_users_final = valid_users_initial.intersection(valid_users_genre_filtered)
        filtered_interactions = filtered_interactions[filtered_interactions['user_id'].isin(valid_users_final)].copy()
        logger.info(f"Users with >={min_interactions} interactions and <=2 genres (final count): {len(valid_users_final)}")
    else:
        logger.info(f"Users with >={min_interactions} interactions (no genre filter applied): {len(valid_users_final)}")

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
        content_weight=config.get('hybrid_content_weight', 0.5),
        mf_weight=config.get('hybrid_mf_weight', 0.5),
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
    
    shared_recommenders = {}
    if 'ContentBased' in config.get('recommender_types_to_run', []):
        system.add_recommender(content_recommender)
        shared_recommenders['content_recommender'] = content_recommender
    if 'CollaborativeFiltering' in config.get('recommender_types_to_run', []):
        system.add_recommender(cf_recommender)
        shared_recommenders['cf_recommender'] = cf_recommender
    if 'MatrixFactorization' in config.get('recommender_types_to_run', []):
        system.add_recommender(mf_recommender)
        shared_recommenders['mf_recommender'] = mf_recommender
    if 'HybridContentMF' in config.get('recommender_types_to_run', []):
        system.add_recommender(hybrid_recommender)
        shared_recommenders['hybrid_recommender'] = hybrid_recommender

    cache_dir = config.get('cache_dir')
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    
    shared_data = {
        'system': system,
        'train_interactions': train_interactions,
        'test_interactions': test_interactions,
        'song_metadata': full_song_metadata,
        'song_features': full_song_features,
        'artist_identification': artist_identification_raw,
        'mappings': {
            'song_to_tier_map': song_to_tier_map,
            'song_to_pop_map': song_to_pop_map,
            'song_to_fam_map': song_to_fam_map,
            'song_to_genres_map': song_to_genres_map,
            'song_to_language_map': song_to_language_map,
            'related_genres_map': related_genres_map,
            'song_to_artist_map': song_to_artist_map
        },
        'content_recommender': content_recommender,
        'cf_recommender': cf_recommender,
        'mf_recommender': mf_recommender,
        'hybrid_recommender': hybrid_recommender
    }
    
    return shared_data

def _load_and_apply_component_tier_weights(hybrid_recommender, component_name, k, config, shared_data):
    """Load and apply optimized tier weights to a component of the HybridRecommender."""
    try:
        db_file = os.path.join(config['optuna_db_path'], f"optuna_study_{component_name}_k{k}.db")
        if os.path.exists(db_file):
            weights, _ = load_optuna_weights(db_file, f"study_{component_name}_k{k}")
            if weights:
                component_map = {
                    'ContentBased': hybrid_recommender.content_recommender,
                    'MatrixFactorization': hybrid_recommender.mf_recommender
                }
                component_recommender = component_map.get(component_name)
                if component_recommender:
                    component_recommender.tier_weights = weights
                    logger.info(f"Applied optimized tier weights to {component_name} component for k={k}: {weights}")
                else:
                    logger.warning(f"Component {component_name} not found in hybrid recommender. Skipping weight application.")
            else:
                logger.warning(f"No optimized weights loaded for {component_name} at k={k}. Using default weights.")
                component_map = {
                    'ContentBased': hybrid_recommender.content_recommender,
                    'MatrixFactorization': hybrid_recommender.mf_recommender
                }
                component_recommender = component_map.get(component_name)
                if component_recommender:
                    component_recommender.tier_weights = config['tier_weights']
                    logger.info(f"Applied default tier weights to {component_name} component for k={k}: {config['tier_weights']}")
        else:
            logger.warning(f"Optuna DB file for {component_name} at k={k} not found. Using default weights.")
            component_map = {
                'ContentBased': hybrid_recommender.content_recommender,
                'MatrixFactorization': hybrid_recommender.mf_recommender
            }
            component_recommender = component_map.get(component_name)
            if component_recommender:
                component_recommender.tier_weights = config['tier_weights']
                logger.info(f"Applied default tier weights to {component_name} component for k={k}: {config['tier_weights']}")
    except Exception as e:
        logger.error(f"Error loading/applying weights for {component_name} at k={k}: {str(e)}")
        component_map = {
            'ContentBased': hybrid_recommender.content_recommender,
            'MatrixFactorization': hybrid_recommender.mf_recommender
        }
        component_recommender = component_map.get(component_name)
        if component_recommender:
            component_recommender.tier_weights = config['tier_weights']
            logger.info(f"Applied default tier weights to {component_name} component for k={k} due to error: {config['tier_weights']}")

def execute_training(config, shared_data):
    """Train the recommendation system based on config."""
    logger.info("--- Starting Training Phase ---")
    
    if config.get('skip_training', False):
        logger.info("Training skipped as per config.")
        return

    system = shared_data['system']
    train_interactions = shared_data['train_interactions']
    song_metadata = shared_data['song_metadata']
    song_features = shared_data['song_features']
    
    logger.info(f"Training with {len(train_interactions)} interactions")
    system.train(
        interactions=train_interactions,
        song_metadata=song_metadata,
        song_features=song_features,
        verbose=config['verbose']
    )
    
    logger.info("--- Training Phase Complete ---")

def execute_optimization(config, shared_data):
    """Run optimization for all recommenders and k-values."""
    logger.info("--- Starting Optimization Phase ---")
    
    if not config.get('run_optimization', False):
        logger.info("Optimization skipped as per config.")
        return {}

    best_weights_per_recommender = {}
    for recommender_name in config.get('recommender_types_to_run', []):
        best_weights_per_recommender[recommender_name] = {}
        for k_val in config['k_values']:
            logger.info(f"Optimizing for {recommender_name} with k={k_val}")
            weights, loss, metrics = optimize_tier_weights(
                recommender_name=recommender_name,
                k=k_val,
                config=config,
                shared_data=shared_data
            )
            best_weights_per_recommender[recommender_name][k_val] = (weights, loss, metrics)
            logger.info(f"Best weights for {recommender_name} at k={k_val}: {weights}")
    
    return best_weights_per_recommender

def perform_evaluation_run(config, shared_data, recommender_key):
    """Helper to evaluate a single recommender for a given k."""
    system = shared_data['system']
    test_interactions = shared_data['test_interactions']
    song_metadata = shared_data['song_metadata']
    
    metrics = {}
    loss = {}
    
    for k_val in config['k_values']:
        try:
            current_metrics = system.evaluate(
                recommender_name=recommender_key,
                interactions=test_interactions,
                song_metadata=song_metadata,
                k=k_val,
                verbose=config['verbose']
            )
            metrics.update(current_metrics)
            
            if recommender_key in config['loss_calculators']:
                loss_calculator = config['loss_calculators'][recommender_key]
                current_loss = loss_calculator.compute_loss(current_metrics, k=k_val)
                loss[k_val] = current_loss
            else:
                loss[k_val] = 0.0
        except Exception as e:
            logger.error(f"Error evaluating {recommender_key} for k={k_val}: {str(e)}")
            for metric_name in ['Precision', 'Genre Precision', 'Language Precision', 'Recall', 'NDCG', 'Hit Rate', 'Emerging Artist Hit Rate', 'Coverage (%)', 'Diversity', 'Novelty', 'Emerging Artist Exposure Index']:
                metrics[metric_name] = metrics.get(metric_name, {})
                metrics[metric_name][k_val] = 0.0
            loss[k_val] = 0.0
    
    return metrics, loss

def execute_evaluation(config, shared_data, best_weights_per_recommender):
    """Run evaluation for all recommenders and k-values."""
    logger.info("\n--- Starting Evaluation Phase ---")
    
    metrics_path = os.path.join(config['saved_data_dir'], f"metrics_u{config['max_users']}.pkl") if config.get('saved_data_dir') else None
    
    final_metrics_per_recommender = {}
    final_loss_per_recommender = {}
    
    recommender_names_from_config = config.get('recommender_types_to_run', [])
    recommender_keys_to_evaluate = []
    
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

            weights_for_this_recommender_k = best_weights_per_recommender.get(recommender_name, {}).get(k_val, (config['tier_weights'], None, None))[0]
            if recommender_name == 'HybridContentMF':
                recommender.content_weight = weights_for_this_recommender_k.get('content_weight', config.get('hybrid_content_weight', 0.5))
                recommender.mf_weight = weights_for_this_recommender_k.get('mf_weight', config.get('hybrid_mf_weight', 0.5))
                logger.info(f"Applied hybrid weights for {recommender_name} at k={k_val}: Content={recommender.content_weight:.2f}, MF={recommender.mf_weight:.2f}")
                _load_and_apply_component_tier_weights(recommender, 'ContentBased', k_val, config, shared_data)
                _load_and_apply_component_tier_weights(recommender, 'MatrixFactorization', k_val, config, shared_data)
            else:
                recommender.tier_weights = weights_for_this_recommender_k
                logger.info(f"Using weights for {recommender_name} at k={k_val}: {recommender.tier_weights}")

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
    for recommender_name in ['ContentBased', 'CollaborativeFiltering', 'MatrixFactorization', 'HybridContentMF']:
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
        'data_path': '/Users/varadkulkarni/Thesis-FaRM/Recommenation-System/System/Finaldata',
        'optuna_db_path': '/Users/varadkulkarni/Thesis-FaRM/Recommenation-System/System/Finaldata/',
        'saved_data_dir': '/Users/varadkulkarni/Thesis-FaRM/Recommenation-System/System/SavedData2',
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
        'recommender_types_to_run': ['HybridContentMF'],
        'cache_dir': os.path.join(project_root, 'System', 'SavedData', 'cache'),
        'tensorboard_log_dir': os.path.join(project_root, 'System', 'TensorBoardLogs'),
    }
    main_run(base_config)