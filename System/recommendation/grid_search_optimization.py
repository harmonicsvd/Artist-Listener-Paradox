import logging
from itertools import product
import pandas as pd
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_weights(weights):
    """Normalize a list of weights to sum to 1.

    Args:
        weights (list): List of weight values.

    Returns:
        list: Normalized weights summing to 1.
    """
    total = sum(weights)
    if total == 0:
        return [1.0 / len(weights)] * len(weights)  # Equal weights if sum is 0
    return [w / total for w in weights]

def evaluate_weights(system, content_recommender, evaluator, train_interactions, test_interactions, k_values, weights_dict, eval_users, k=2):
    """Evaluate the recommendation system with given tier weights and return the objective loss and metrics for k.

    Args:
        system: Initialized RecommendationSystem instance.
        content_recommender: Trained ContentBasedRecommender instance.
        evaluator: RecommendationEvaluator instance.
        train_interactions: Training interactions DataFrame.
        test_interactions: Test interactions DataFrame.
        k_values: List of k values for evaluation.
        weights_dict: Dictionary of tier weights.
        eval_users: List of user IDs to evaluate.
        k: The k value to optimize for (default is 2).

    Returns:
        tuple: (loss for k, metrics) or (float('inf'), {}) if invalid.
    """
    try:
        # Update recommender weights
        content_recommender.tier_weights = weights_dict
        metrics = {}
        tier_diversity_by_k = {}
        gini_by_k = {}
        recs_by_k = {}

        for k_val in k_values:
            eval_recs = []
            system.artist_exposure = {}
            valid_user_count = 0
            for user_id in eval_users:
                if user_id in content_recommender.user_profiles or train_interactions['user_id'].isin([user_id]).any():
                    recs = content_recommender.recommend(
                        user_id=user_id,
                        n=k_val,
                        exclude_listened=False,
                        include_metadata=True,
                        testing_mode=True,
                        user_interactions=train_interactions
                    )
                    if not recs.empty and 'user_id' in recs.columns and 'song_id' in recs.columns:
                        invalid_songs = set(recs['song_id']) - set(content_recommender.item_ids)
                        if invalid_songs:
                            logger.warning(f"Invalid song_ids for user {user_id}: {invalid_songs}")
                        eval_recs.append(recs)
                        valid_user_count += 1
                        if 'artist_id' in recs.columns:
                            system.track_exposure(recs, content_recommender.song_to_artist)
                        else:
                            logger.warning(f"No artist_id in recommendations for user {user_id}")
                    else:
                        logger.debug(f"No valid recommendations for user {user_id} at k={k_val}")

            logger.debug(f"Users with valid recommendations for k={k_val}: {valid_user_count}/{len(eval_users)}")
            
            if not eval_recs:
                logger.error(f"No recommendations generated for k={k_val}")
                return float('inf'), {}

            eval_recs = pd.concat(eval_recs, ignore_index=True)
            logger.debug(f"Generated {len(eval_recs)} recommendations for k={k_val}")
            
            exposure_result = system.analyze_exposure_distribution(song_metadata=content_recommender.song_metadata)
            exposure_df = exposure_result.get('exposure_df', pd.DataFrame())
            if exposure_df.empty:
                logger.warning(f"No exposure data for k={k_val}")
                tier_diversity_by_k[k_val] = 0.0
                gini_by_k[k_val] = 0.0
            else:
                tier_diversity_by_k[k_val] = exposure_result.get('tier_diversity', 0.0)
                gini_by_k[k_val] = exposure_result.get('gini_coefficient', 0.0)

            recs_by_k[k_val] = eval_recs
            metrics.update(evaluator.evaluate(recommendations=eval_recs, test_interactions=test_interactions))

        if not metrics or not recs_by_k:
            logger.error(f"No metrics or recommendations generated: metrics={metrics}, recs_by_k={recs_by_k}")
            return float('inf'), {}

        logger.debug(f"Metrics for loss calculation: {metrics}")
        from System.recommendation.objective_loss import ObjectiveLossCalculator
        loss_calculator = ObjectiveLossCalculator()
        loss = loss_calculator.compute_objective_loss(
            metrics=metrics,
            k_values=k_values,
            tier_diversity_by_k=tier_diversity_by_k,
            gini_by_k=gini_by_k
        )

        if loss is None or not isinstance(loss, dict) or k not in loss:
            logger.error(f"Invalid loss computed: {loss}")
            return float('inf'), {}

        logger.debug(f"Loss for k={k}: {loss[k]:.4f}")
        return loss[k], metrics
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        return float('inf'), {}

def optimize_tier_weights(config, system, content_recommender, evaluator, train_interactions, test_interactions, eval_users, tiers, weight_options, k=2):
    """Perform grid search to find optimal tier weights that minimize objective loss for k=2.

    Args:
        config (dict): Configuration dictionary.
        system: Initialized RecommendationSystem instance.
        content_recommender: Trained ContentBasedRecommender instance.
        evaluator: RecommendationEvaluator instance.
        train_interactions: Training interactions DataFrame.
        test_interactions: Test interactions DataFrame.
        eval_users: List of user IDs to evaluate.
        tiers (list): List of tier names.
        weight_options (list): List of weight values to test.
        k (int): The k value to optimize for (default is 2).

    Returns:
        tuple: (best_weights_dict, best_loss, best_metrics)
    """
    # Grid search
    combinations = list(product(weight_options, repeat=len(tiers)))
    best_loss = float('inf')
    best_weights = None
    best_metrics = {}

    logger.info(f"Total combinations to test: {len(combinations)}")
    
    for combo in tqdm(combinations, desc="Grid Search Progress"):
        normalized_weights = normalize_weights(combo)
        weights_dict = {tier: weight for tier, weight in zip(tiers, normalized_weights)}
        logger.debug(f"Testing weights: {weights_dict}")
        
        loss, metrics = evaluate_weights(
            system=system,
            content_recommender=content_recommender,
            evaluator=evaluator,
            train_interactions=train_interactions,
            test_interactions=test_interactions,
            k_values=config['k_values'],
            weights_dict=weights_dict,
            eval_users=eval_users,
            k=k
        )
        
        if loss < best_loss:
            best_loss = loss
            best_weights = weights_dict
            best_metrics = metrics
            logger.info(f"New best loss: {loss:.4f} with weights: {weights_dict}")

    if best_weights is None:
        logger.error("No valid weight combinations found")
        return None, float('inf'), {}

    print("\nBest Tier Weights Found:")
    for tier, weight in best_weights.items():
        print(f"{tier}: {weight:.4f}")
    print(f"Best Objective Loss for k=2: {best_loss:.4f}")

    return best_weights, best_loss, best_metrics