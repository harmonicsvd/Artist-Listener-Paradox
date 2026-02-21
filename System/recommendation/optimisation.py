import logging
import numpy as np
from itertools import product
from tqdm.auto import tqdm # Use tqdm.auto for intelligent import
import optuna
from optuna.samplers import TPESampler
import pandas as pd
import os
import pickle
import tensorflow as tf # Import TensorFlow for TensorBoard logging

from System.recommendation.recommendation_system import RecommendationSystem

logger = logging.getLogger(__name__)

def normalize_weights(weights):
    """Normalize weights to sum to 1."""
    total = sum(weights)
    if total == 0:
        # Handle case where all weights are zero, distribute evenly
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]

def evaluate_user_weights(user_id, config, recommender, train_interactions, k):
    """Helper function to generate recommendations for a single user with pre-initialized recommender."""
    try:
        recs = recommender.recommend(
            user_id=user_id,
            n=k,
            exclude_listened=False,
            include_metadata=True,
            testing_mode=config['verbose'],
            user_interactions=train_interactions
        )
        return recs if not recs.empty else None
    except Exception as e:
        # Use recommender.name if it exists, otherwise default to a string representation
        recommender_name = getattr(recommender, 'name', str(type(recommender).__name__))
        logger.error(f"Error generating recommendations for user {user_id} with recommender {recommender_name}: {str(e)}")
        return None

def evaluate_weights(config, system, recommender, evaluator, train_interactions, test_interactions, weights_to_apply, eval_users, k, is_hybrid_optimization=False):
    """
    Evaluate objective loss for a given set of weights sequentially.
    This function is generic for any recommender that uses tier_weights or hybrid weights.
    Returns objective_loss, metrics, listener_satisfaction, artist_satisfaction, gini_coefficient.
    """
    try:
        system.artist_exposure = {} # Reset exposure for this evaluation run
        
        # Apply the current set of weights to the recommender
        if is_hybrid_optimization:
            recommender.content_weight = weights_to_apply.get('content_weight', 0.5)
            recommender.mf_weight = weights_to_apply.get('mf_weight', 0.5)
            logger.debug(f"Applying hybrid weights: Content={recommender.content_weight:.2f}, MF={recommender.mf_weight:.2f}")
        else:
            recommender.tier_weights = weights_to_apply
            logger.debug(f"Applying tier weights: {recommender.tier_weights}")

        eval_recs = []
        # Progress bar for generating recs for users is ALWAYS visible (disable=False)
        for user_id in tqdm(eval_users, desc=f"Generating recs for k={k} with weights", leave=False, disable=False):
            recs = evaluate_user_weights(user_id, config, recommender, train_interactions, k)
            if recs is not None:
                eval_recs.append(recs)

        if not eval_recs:
            logger.warning(f"No recommendations generated for k={k} with weights {weights_to_apply}. Returning a high loss and empty metrics.")
            return float('inf'), {}, 0.0, 0.0, 1.0 # High loss, empty metrics, default exposure values

        eval_recs_df = pd.concat(eval_recs, ignore_index=True)
        
        # Track artist exposure with the correct song_to_artist mapping from recommender
        system.track_exposure(eval_recs_df, recommender.song_to_artist)

        metrics = evaluator.evaluate(recommendations=eval_recs_df, test_interactions=test_interactions)
        
        # Recalculate exposure metrics (tier_diversity and gini_coefficient are still needed from here)
        # Note: 'Emerging Artist Exposure Index' is already in 'metrics' from evaluator.evaluate()
        exposure_result = system.analyze_exposure_distribution(
            song_metadata=recommender.song_metadata # Use recommender's song_metadata
        )
        tier_diversity = exposure_result.get('tier_diversity', 0.0)
        gini_coefficient = exposure_result.get('gini_coefficient', 0.0)

        # Ensure tier_diversity and gini_coefficient are also passed for logging if needed
        # They are separate from the 'metrics' object for the loss calculator, as per objective_loss.py's method signature.

        # Get the objective loss calculator for this recommender
        recommender_name_for_loss = getattr(recommender, 'name', str(type(recommender).__name__))
        loss_calculator = config['loss_calculators'][recommender_name_for_loss]
        
        # The objective loss calculator expects metrics to be a dict of dicts,
        # with the inner dict keyed by k-value.
        formatted_metrics_for_loss_calc = {metric_name: {k_val: values.get(k_val, 0.0)} for metric_name, values in metrics.items() for k_val in [k]}

        # --- Debugging LS, AS, Gini inputs ---
        logger.debug(f"Input to ObjectiveLossCalculator for k={k}:")
        logger.debug(f"  metrics: {formatted_metrics_for_loss_calc}")
        logger.debug(f"  tier_diversity_by_k: {{ {k}: {tier_diversity:.4f} }}")
        logger.debug(f"  gini_by_k: {{ {k}: {gini_coefficient:.4f} }}")
        # --- End Debugging ---

        objective_loss_dict, ls_dict, as_dict = loss_calculator.compute_objective_loss(
            metrics=formatted_metrics_for_loss_calc, # This metrics object now contains the correct Exposure Index
            k_values=[k],
            tier_diversity_by_k={k: tier_diversity},
            gini_by_k={k: gini_coefficient}
        )
        objective_loss = objective_loss_dict.get(k, float('inf'))
        listener_satisfaction_overall = ls_dict.get(k, 0.0)
        artist_satisfaction_overall = as_dict.get(k, 0.0)
        
        logger.info(f"Objective Loss for k={k}: {objective_loss:.4f} (LS={listener_satisfaction_overall:.4f}, AS={artist_satisfaction_overall:.4f}, Gini={gini_coefficient:.4f})")
        
        # Return metrics which includes the correctly calculated Emerging Artist Exposure Index
        return objective_loss, metrics, listener_satisfaction_overall, artist_satisfaction_overall, gini_coefficient

    except Exception as e:
        recommender_name = getattr(recommender, 'name', str(type(recommender).__name__))
        logger.error(f"Error during evaluation of weights {weights_to_apply} for recommender {recommender_name}: {str(e)}")
        return float('inf'), {}, 0.0, 0.0, 1.0 # Return high loss on error

def optimize_tier_weights(
    config,
    system,
    recommender,
    evaluator,
    train_interactions,
    test_interactions,
    eval_users,
    tiers,
    recommender_name, # This recommender_name is used for logging and TensorBoard path
    weight_options,
    optimization_method,
    k,
    n_trials, # This parameter is now correctly defined here
    is_hybrid_optimization: bool = False
):
    """
    Optimizes tier weights (or hybrid weights) using Optuna.
    """
    study_name = f"study_{recommender_name}_k{k}"
    # Prepend 'optuna_' to the database file name for consistency with existing files
    db_path = os.path.join(config['optuna_db_path'], f"optuna_{study_name}.db")
    storage = f"sqlite:///{db_path}"

    # Create TensorBoard writer
    log_dir = os.path.join(config['tensorboard_log_dir'], recommender_name, f"k_{k}")
    os.makedirs(log_dir, exist_ok=True) # Ensure log directory exists
    writer = tf.summary.create_file_writer(log_dir)
    logger.info(f"TensorBoard logs will be written to: {log_dir}")

    best_loss = float('inf')
    best_weights = {} # Initialize as empty dict, will be populated by Optuna
    best_metrics = {}

    def objective(trial: optuna.Trial):
        weights = {}
        if is_hybrid_optimization:
            # For hybrid, optimize content_weight and mf_weight
            # Ensure they sum to 1 by making one dependent on the other
            content_weight = trial.suggest_float('content_weight', 0.01, 0.99)
            mf_weight = 1.0 - content_weight
            weights = {'content_weight': content_weight, 'mf_weight': mf_weight}
            logger.debug(f"Trial {trial.number}: Hybrid weights proposed: {weights}")
        else:
            # For other recommenders, optimize tier weights
            for i, tier in enumerate(tiers):
                if optimization_method == 'grid':
                    weights[tier] = trial.suggest_categorical(f'w_{tier}', weight_options)
                else: # bayesian, etc.
                    weights[tier] = trial.suggest_float(f'w_{tier}', 0.01, 1.0) # Broad range, Optuna will refine

            # Normalize weights to sum to 1 for non-hybrid tier weights
            total_weight = sum(weights.values())
            if total_weight == 0: # Avoid division by zero
                normalized_weights = {tier: 1.0 / len(weights) for tier in weights}
            else:
                normalized_weights = {tier: w / total_weight for tier, w in weights.items()}
            weights = normalized_weights # Use normalized weights for evaluation
            logger.debug(f"Trial {trial.number}: Tier weights proposed: {weights}")
        
        # Evaluate these weights using the dedicated evaluation function
        current_loss, current_metrics, ls_score, as_score, gini_score = evaluate_weights(
            config, system, recommender, evaluator, train_interactions, test_interactions, weights, eval_users, k, is_hybrid_optimization
        )
        
        # Log metrics to TensorBoard for this trial
        with writer.as_default():
            tf.summary.scalar("objective_loss", current_loss, step=trial.number)
            
            for metric_name, values in current_metrics.items():
                # Ensure the metric exists for the current k-value before logging
                if k in values:
                    tf.summary.scalar(f"metrics/{metric_name}", values[k], step=trial.number)
            tf.summary.scalar("metrics/listener_satisfaction", ls_score, step=trial.number)
            tf.summary.scalar("metrics/artist_satisfaction", as_score, step=trial.number)
            tf.summary.scalar("metrics/gini_coefficient", gini_score, step=trial.number)
            tf.summary.text("trial_weights", str(weights), step=trial.number) # Renamed to avoid confusion

        # Store best weights for this trial as user attribute
        trial.set_user_attr('best_weights_for_trial', weights)
        trial.set_user_attr('best_metrics_for_trial', current_metrics)
        trial.set_user_attr('ls_for_trial', ls_score)
        trial.set_user_attr('as_for_trial', as_score)
        trial.set_user_attr('gini_for_trial', gini_score)

        return current_loss

    try:
        # Load or create the Optuna study
        # Catching KeyError directly because the traceback indicates Optuna raises KeyError for non-existent studies.
        study = optuna.load_study(study_name=study_name, storage=storage)
        logger.info(f"Loaded existing study '{study_name}' from {db_path}")
    except KeyError: # Catch KeyError, which is raised if the study_name doesn't exist in the DB
        logger.info(f"Creating new study '{study_name}' at '{storage}' for {recommender_name} (Study not found).")
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            storage=storage,
            study_name=study_name
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading or creating study {study_name}: {e}")
        raise # Re-raise unexpected exceptions


    # Optimize with a progress bar for the trials, set to always display
    logger.info(f"Starting Optuna optimization for {recommender_name} k={k} with {n_trials} trials.")
    with tqdm(total=n_trials, desc=f"Optimizing {recommender_name} (k={k})", disable=False) as pbar: # Changed disable to False
        for i in range(n_trials):
            # Run one trial at a time to update tqdm and log progress
            study.optimize(objective, n_trials=1, show_progress_bar=False) # Disable Optuna's internal progress bar

            # Update progress bar description with current best trial info
            if study.best_trial:
                best_trial_so_far = study.best_trial
                current_best_loss = best_trial_so_far.value
                
                # Get the weights for display based on whether it's hybrid optimization
                weights_for_display = {}
                if is_hybrid_optimization:
                    weights_for_display['content_weight'] = f"{best_trial_so_far.user_attrs.get('best_weights_for_trial', {}).get('content_weight', 0.0):.2f}"
                    weights_for_display['mf_weight'] = f"{best_trial_so_far.user_attrs.get('best_weights_for_trial', {}).get('mf_weight', 0.0):.2f}"
                else:
                    weights_for_display = {
                        tier: f"{best_trial_so_far.user_attrs.get('best_weights_for_trial', {}).get(tier, 0.0):.2f}"
                        for tier in tiers # Use the 'tiers' list to ensure consistent order/keys
                    }
                
                # Update progress bar description to show trial number out of total, and best loss
                pbar.set_description(f"Optimizing {recommender_name} (k={k}) | Trial {i+1}/{n_trials} | Best Loss: {current_best_loss:.4f} | Weights: {weights_for_display}")

                # Log the best weights found so far to TensorBoard
                with writer.as_default():
                    tf.summary.text("best_weights_so_far", str(weights_for_display), step=i+1)
                    writer.flush() # Ensure logs are written to disk immediately

            pbar.update(1)

    best_trial = study.best_trial
    best_loss = best_trial.value
    # Retrieve the best weights and metrics from user_attrs
    best_weights = best_trial.user_attrs.get('best_weights_for_trial', best_trial.params) 
    best_metrics = best_trial.user_attrs.get('best_metrics_for_trial', {})
    best_ls_overall = best_trial.user_attrs.get('ls_for_trial', 0.0)
    best_as_overall = best_trial.user_attrs.get('as_for_trial', 0.0)
    best_gini_overall = best_trial.user_attrs.get('gini_for_trial', 1.0) # Gini can be 1.0 for very unequal distributions

    logger.info(f"Bayesian optimization for {recommender_name} k={k} complete. Best loss: {best_loss:.4f}, Best weights: {best_weights}")
    logger.info(f"Best LS: {best_ls_overall:.4f}, Best AS: {best_as_overall:.4f}, Best Gini: {best_gini_overall:.4f}")

    # Log final best results to TensorBoard (can be viewed under 'HPARAMS' or custom dashboards)
    with writer.as_default():
        tf.summary.scalar("final_best_loss_overall", best_loss, step=n_trials)
        for metric_name, values in best_metrics.items():
            if k in values: # Log for the specific k-value
                tf.summary.scalar(f"final_best_metrics_overall/{metric_name}", values[k], step=n_trials)
        tf.summary.scalar("final_best_metrics_overall/listener_satisfaction", best_ls_overall, step=n_trials)
        tf.summary.scalar("final_best_metrics_overall/artist_satisfaction", best_as_overall, step=n_trials)
        tf.summary.scalar("final_best_metrics_overall/gini_coefficient", best_gini_overall, step=n_trials)
        tf.summary.text('Best_Results/Weights_Final', str(best_weights), step=n_trials) # Renamed for clarity
        writer.flush()
    writer.close() # Close the writer after optimization for this study

    # This function should return exactly 3 values as expected by run_system.py
    return best_weights, best_loss, best_metrics
