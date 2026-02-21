#!/usr/bin/env python3
"""
Unified entry-point MainSystem.py for your FaRM music recommendation platform.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import tensorflow as tf

# Add your project root to sys.path to allow package imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from System.recommendation.recommendation_system import RecommendationSystem
from System.recommendation.content_based import ContentBasedRecommender
from System.recommendation.collaborative_filtering import CollaborativeFilteringRecommender
from System.recommendation.matrix_factorization import MatrixFactorizationRecommender
from System.recommendation.hybrid_recommender import HybridRecommender
from System.recommendation.objective_loss import ObjectiveLossCalculator
from config_loader import load_config
from run_system4 import (
    initialize_shared_resources,
    execute_training,
    execute_optimization,
    execute_evaluation,
    execute_recommendation_comparison,
    load_optuna_weights, _load_and_apply_component_tier_weights
)


logger = logging.getLogger("FaRMUnifiedRunner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified runner for FaRM recommendation system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        required=True,
        help="YAML/JSON/TOML file containing the base_config dictionary",
    )
    p.add_argument(
        "--recs",
        nargs="+",
        choices=[
            "ContentBased",
            "CollaborativeFiltering",
            "MatrixFactorization",
            "HybridContentMF",
        ],
        required=True,
        help="Recommenders to activate in this run",
    )
    p.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[5],
        help="k-values to evaluate (e.g. 5 10 20)",
    )
    p.add_argument(
        "--no-optim",
        action="store_true",
        help="Skip Optuna optimisation phase",
    )
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and optimisation; run evaluation with cached artifacts",
    )
    return p.parse_args()


def enrich_config(base_conf: dict, args: argparse.Namespace) -> dict:
    base_conf = base_conf.copy()
    base_conf["k_values"] = args.k
    base_conf["recommender_types_to_run"] = args.recs

    # Honor CLI switches
    if args.no_optim:
        base_conf["run_optimization"] = False
    if args.eval_only:
        base_conf["run_optimization"] = False
        base_conf["skip_training"] = True

    # Timestamp TensorBoard logs
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    base_conf["tensorboard_log_dir"] = os.path.join(base_conf["tensorboard_log_dir"], f"run_{stamp}")

    return base_conf


def main():
    args = cli()
    base_cfg = load_config(args.config)
    cfg = enrich_config(base_cfg, args)

    # Instantiate loss calculators from dict config
    if 'loss_calculators' in cfg:
        from System.recommendation.objective_loss import ObjectiveLossCalculator
        cfg['loss_calculators'] = {
            k: ObjectiveLossCalculator(**v) for k, v in cfg['loss_calculators'].items()
        }

    logger.info(f"Active recommenders: {cfg['recommender_types_to_run']}")
    logger.info(f"k-values: {cfg['k_values']}")
    logger.info(f"Optimization enabled: {cfg.get('run_optimization', True)}")

    # Setup TensorBoard logging
    tb_root = Path(cfg["tensorboard_log_dir"])
    tb_root.mkdir(parents=True, exist_ok=True)
    tb_writer = tf.summary.create_file_writer(str(tb_root / "overall"))
    with tb_writer.as_default():
        tf.summary.text("run_state", "initializing", step=0)

    # Initialize shared data & recommenders
    shared_data = initialize_shared_resources(cfg)

    # Prepare evaluation user list
    test_users = shared_data['test_interactions']['user_id'].unique()
    train_user_counts = shared_data['train_interactions']['user_id'].value_counts()
    eval_users = train_user_counts[train_user_counts.index.isin(test_users)].index.tolist()
    if not eval_users:
        logger.error("No evaluation users found; skipping evaluation & comparison phases.")
    else:
        cfg['eval_users'] = eval_users

    # Train if enabled
    if not cfg.get('skip_training', False):
        execute_training(cfg, shared_data)

    # Optimize or load weights
    best_weights = {}
    if cfg.get('run_optimization', False):
        best_weights = execute_optimization(cfg, shared_data)
    elif cfg.get('load_optimized_weights', False):
        # Optionally load weights from disk for all recommenders and k's
        best_weights = {}
        for rec in cfg['recommender_types_to_run']:
            best_weights[rec] = {}
            for k in cfg['k_values']:
                db_file = os.path.join(cfg['optuna_db_path'], f"optuna_study_{rec}_k{k}.db")
                if os.path.exists(db_file):
                    loaded_weights, _ = load_optuna_weights(db_file, f"study_{rec}_k{k}")
                    if loaded_weights:
                        best_weights[rec][k] = (loaded_weights, None, None)
                        logger.info(f"Loaded optimized weights for {rec} at k={k}")
                    else:
                        best_weights[rec][k] = (cfg['tier_weights'], None, None)
                        logger.warning(f"Failed to load weights for {rec} at k={k}, using defaults.")
                else:
                    best_weights[rec][k] = (cfg['tier_weights'], None, None)
                    logger.warning(f"Weight DB missing for {rec} at k={k}, using defaults.")
    else:
        # Use default tier weights for all recommenders/k if no optimization or load
        for rec in cfg['recommender_types_to_run']:
            best_weights[rec] = {k: (cfg['tier_weights'], None, None) for k in cfg['k_values']}

    # Map recommender names to shared_data keys (= recommender instances)
    recommender_key_map = {
        'ContentBased': 'content_recommender',
        'CollaborativeFiltering': 'cf_recommender',
        'MatrixFactorization': 'mf_recommender',
        'HybridContentMF': 'hybrid_recommender',
    }

    # APPLY optimized weights BEFORE generating any recommendations
    max_k = max(cfg['k_values'])
    for rec_name in cfg['recommender_types_to_run']:
        rec_key = recommender_key_map.get(rec_name)
        rec_obj = shared_data.get(rec_key)
        if rec_obj is None:
            logger.warning(f"Recommender object for {rec_name} not found. Skipping weight application.")
            continue

        if rec_name in best_weights and max_k in best_weights[rec_name]:
            weights_for_k = best_weights[rec_name][max_k][0]
            if rec_name == 'HybridContentMF':
                # Hybrid recommender has component weights plus tier_weights
                rec_obj.content_weight = weights_for_k.get("content_weight", 0.5)
                rec_obj.mf_weight = weights_for_k.get("mf_weight", 0.5)
                # Apply tier weights also to components if you have helper loaded
                _load_and_apply_component_tier_weights(rec_obj, 'ContentBased', max_k, cfg, shared_data)
                _load_and_apply_component_tier_weights(rec_obj, 'MatrixFactorization', max_k, cfg, shared_data)
                logger.info(f"Applied hybrid weights for {rec_name} at k={max_k}")
            else:
                rec_obj.tier_weights = weights_for_k
                logger.info(f"Applied tier weights for {rec_name} at k={max_k}")
        else:
            rec_obj.tier_weights = cfg['tier_weights']
            logger.warning(f"Using default tier weights for {rec_name} at k={max_k} (no optimized weights found)")

    # Run evaluation phase
    execute_evaluation(cfg, shared_data, best_weights)

    # Run recommendation comparison with recommender objects (not names)
    def recommendation_comparison_with_objects(cfg_inner, shared_data_inner, best_wts):
        for rec_name in cfg_inner['recommender_types_to_run']:
            rec_key = recommender_key_map.get(rec_name)
            rec_obj = shared_data_inner.get(rec_key)
            if rec_obj is None:
                logger.warning(f"Recommender {rec_name} object not found for comparison. Skipping.")
                continue
            try:
                execute_recommendation_comparison(cfg_inner, shared_data_inner, best_wts)
            except Exception as e:
                logger.error(f"Error comparing recommendations for {rec_name}: {e}")

    if 'eval_users' in cfg and cfg['eval_users']:
        recommendation_comparison_with_objects(cfg, shared_data, best_weights)
    else:
        logger.warning("No evaluation users - skipping recommendation comparison.")

    logger.info("Run finished successfully.")

    with tb_writer.as_default():
        tf.summary.text("run_state", "finished", step=1)
        tf.summary.scalar("end_timestamp", float(datetime.now().timestamp()), step=1)

    tb_writer.flush()
    tb_writer.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Run interrupted by user.")
        sys.exit(0)  # Exit cleanly on Ctrl+C
        
        """
        python run_system_unified.py \
    --config  configs/base_local.yml \
    --recs    HybridContentMF \
    --k       5 \
    --eval-only
    
    """