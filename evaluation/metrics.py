# ==========================================
# FILE: evaluation/metrics.py
# ==========================================

import numpy as np
import pandas as pd
import json
import datetime
from pathlib import Path

def calculateNDCG(relevance_scores, k=10):
    """
    Calculate Normalized Discounted Cumulative Gain.

    Parameters:
    - relevance_scores: List of relevance scores (1-5) for each recommendation
    - k: Number of recommendations to consider (default: 10)

    Returns:
    - NDCG score between 0 and 1
    """
    if not relevance_scores:
        return 0.0

    # Calculate DCG (Discounted Cumulative Gain)
    dcg = relevance_scores[0]
    for i in range(1, min(k, len(relevance_scores))):
        # Use log base 2 for position discount
        dcg += relevance_scores[i] / np.log2(i + 2)

    # Calculate IDCG (Ideal DCG - relevance scores sorted in descending order)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = ideal_scores[0]
    for i in range(1, min(k, len(ideal_scores))):
        idcg += ideal_scores[i] / np.log2(i + 2)

    # Avoid division by zero
    if idcg == 0:
        return 0.0

    # Normalize DCG
    return dcg / idcg

def calculateRelevanceScore(ratings):
    """
    Calculate average relevance score.

    Parameters:
    - ratings: List of user ratings (1-5) for recommendations

    Returns:
    - Average relevance score between 1 and 5
    """
    if not ratings:
        return 0.0

    # Simple average of all ratings
    return sum(ratings) / len(ratings)

def calculateGenreCoverage(user_interests, recommended_genres):
    """
    Calculate genre coverage percentage.

    Parameters:
    - user_interests: List of genres the user is interested in
    - recommended_genres: List of genres in the recommended games

    Returns:
    - Percentage of user interests covered by recommendations (0-100)
    """
    if not user_interests:
        return 0.0

    # Convert lists to sets for intersection operation
    user_interests_set = set(user_interests)
    recommended_genres_set = set(recommended_genres)

    # Find intersection of user interests and recommended genres
    covered_genres = user_interests_set.intersection(recommended_genres_set)

    # Calculate percentage coverage
    return (len(covered_genres) / len(user_interests_set)) * 100

def calculatePrecisionAtK(relevance_scores, k=10, threshold=4):
    """
    Calculate Precision@K - fraction of top-k recommendations that are relevant.

    Parameters:
    - relevance_scores: List of relevance scores (1-5)
    - k: Number of top recommendations to consider
    - threshold: Minimum score to consider as relevant (default: 4)

    Returns:
    - Precision@K score between 0 and 1
    """
    if not relevance_scores or k <= 0:
        return 0.0

    top_k_scores = relevance_scores[:k]
    relevant_count = sum(1 for score in top_k_scores if score >= threshold)

    return relevant_count / len(top_k_scores)

def calculateRecallAtK(relevance_scores, k=10, threshold=4):
    """
    Calculate Recall@K - fraction of all relevant items that appear in top-k.

    Parameters:
    - relevance_scores: List of relevance scores (1-5)
    - k: Number of top recommendations to consider
    - threshold: Minimum score to consider as relevant (default: 4)

    Returns:
    - Recall@K score between 0 and 1
    """
    if not relevance_scores:
        return 0.0

    total_relevant = sum(1 for score in relevance_scores if score >= threshold)
    if total_relevant == 0:
        return 0.0

    top_k_scores = relevance_scores[:k]
    relevant_in_top_k = sum(1 for score in top_k_scores if score >= threshold)

    return relevant_in_top_k / total_relevant

def calculateDiversityScore(recommended_genres):
    """
    Calculate diversity score based on genre distribution.

    Parameters:
    - recommended_genres: List of genres in recommended games

    Returns:
    - Diversity score between 0 and 1 (higher = more diverse)
    """
    if not recommended_genres:
        return 0.0

    # Count genre frequencies
    genre_counts = {}
    for genre in recommended_genres:
        genre_counts[genre] = genre_counts.get(genre, 0) + 1

    # Calculate Shannon entropy for diversity
    total_games = len(recommended_genres)
    entropy = 0.0

    for count in genre_counts.values():
        probability = count / total_games
        if probability > 0:
            entropy -= probability * np.log2(probability)

    # Normalize by maximum possible entropy
    max_entropy = np.log2(len(genre_counts)) if len(genre_counts) > 1 else 1

    return entropy / max_entropy if max_entropy > 0 else 0.0

def updateUserProfileWeights(user_profile, games_df, ratings, unique_tags_df):
    """
    Legacy function for backward compatibility.
    Now redirects to the enhanced 3-tier system.
    """
    # Import here to avoid circular imports
    from recommender.model import updateUserProfileWith3TierResistance

    return updateUserProfileWith3TierResistance(
        user_profile, games_df, ratings, unique_tags_df
    )

def extractUserGenres(user_profile):
    """
    Extract the user's genres of interest from their profile.

    Parameters:
    - user_profile: User profile DataFrame with tags and weights

    Returns:
    - List of tags/genres the user is interested in (weight > 0)
    """
    # Get tags with non-zero weights
    user_genres = user_profile[user_profile['tag_count'] > 0]['tag'].tolist()
    return user_genres

def extractRecommendedGenres(recommendations, games_df):
    """
    Extract the genres from the recommended games.

    Parameters:
    - recommendations: DataFrame of recommended games
    - games_df: DataFrame with game information

    Returns:
    - List of all genres present in the recommended games
    """
    all_genres = []

    for _, game in recommendations.iterrows():
        # Find the game in games_df
        game_row = games_df[games_df['id'] == game['app_id']]

        if not game_row.empty:
            # Extract tags from the game
            game_tags_str = game_row['tags'].iloc[0]

            # Convert from string representation to actual list
            if isinstance(game_tags_str, str):
                # Remove brackets, quotes and whitespace
                game_tags_str = game_tags_str.replace('[', '').replace(']', '').replace('"', '').replace("'", "")
                game_tags = [tag.strip() for tag in game_tags_str.split(',') if tag.strip()]
                all_genres.extend(game_tags)

    # Remove duplicates
    return list(set(all_genres))

def calculateUserSatisfactionScore(ratings, weights=None):
    """
    Calculate weighted user satisfaction score.

    Parameters:
    - ratings: List of user ratings (1-5)
    - weights: Optional list of weights for each rating (e.g., position weights)

    Returns:
    - Weighted satisfaction score between 0 and 1
    """
    if not ratings:
        return 0.0

    if weights is None:
        # Default equal weights
        weights = [1.0] * len(ratings)

    if len(ratings) != len(weights):
        # If weights don't match, use equal weights
        weights = [1.0] * len(ratings)

    # Calculate weighted average
    weighted_sum = sum(rating * weight for rating, weight in zip(ratings, weights))
    total_weight = sum(weights)

    if total_weight == 0:
        return 0.0

    # Normalize to 0-1 scale (ratings are 1-5)
    avg_rating = weighted_sum / total_weight
    return (avg_rating - 1) / 4  # Convert from 1-5 scale to 0-1 scale

def calculateEngagementMetrics(ratings):
    """
    Calculate various engagement metrics from user ratings.

    Parameters:
    - ratings: List of user ratings (1-5)

    Returns:
    - Dictionary with engagement metrics
    """
    if not ratings:
        return {
            'total_ratings': 0,
            'positive_ratio': 0.0,
            'negative_ratio': 0.0,
            'neutral_ratio': 0.0,
            'average_rating': 0.0,
            'rating_variance': 0.0
        }

    total_ratings = len(ratings)
    positive_count = sum(1 for r in ratings if r >= 4)
    negative_count = sum(1 for r in ratings if r <= 2)
    neutral_count = sum(1 for r in ratings if r == 3)

    return {
        'total_ratings': total_ratings,
        'positive_ratio': positive_count / total_ratings,
        'negative_ratio': negative_count / total_ratings,
        'neutral_ratio': neutral_count / total_ratings,
        'average_rating': sum(ratings) / total_ratings,
        'rating_variance': np.var(ratings)
    }

def saveUserEvaluation(user_id, input_method, selected_tags, recommendations, ratings, metrics):
    """
    Save user evaluation data to JSON file.

    Parameters:
    - user_id: User identifier (can be None for anonymous)
    - input_method: Method used for input ('steam_id' or 'category')
    - selected_tags: List of tags/genres selected by user
    - recommendations: DataFrame of recommended games
    - ratings: Dictionary mapping game IDs to ratings
    - metrics: Dictionary containing evaluation metrics

    Returns:
    - Path to the saved file
    """
    # Create directory if it doesn't exist
    data_dir = Path("user_evaluations")
    data_dir.mkdir(exist_ok=True)

    # Create a unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if user_id:
        filename = f"user_{user_id}_{timestamp}.json"
    else:
        filename = f"anonymous_user_{timestamp}.json"

    # Prepare data to save
    data = {
        "timestamp": timestamp,
        "user_id": user_id,
        "input_method": input_method,
        "selected_tags": selected_tags,
        "recommendations": recommendations.to_dict('records') if isinstance(recommendations, pd.DataFrame) else recommendations,
        "ratings": ratings,
        "metrics": metrics,
        "system_version": "3-tier-enhanced"
    }

    # Save to file
    file_path = data_dir / filename
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4, default=str)  # default=str handles datetime objects
        return file_path
    except Exception as e:
        print(f"Error saving evaluation data: {e}")
        return None

def loadUserEvaluation(file_path):
    """
    Load user evaluation data from JSON file.

    Parameters:
    - file_path: Path to the evaluation file

    Returns:
    - Dictionary with evaluation data or None if error
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading evaluation data: {e}")
        return None

def analyzeEvaluationTrends(evaluation_dir="user_evaluations"):
    """
    Analyze trends across multiple user evaluations.

    Parameters:
    - evaluation_dir: Directory containing evaluation files

    Returns:
    - Dictionary with trend analysis
    """
    eval_dir = Path(evaluation_dir)
    if not eval_dir.exists():
        return {"error": "Evaluation directory not found"}

    evaluations = []
    for file_path in eval_dir.glob("*.json"):
        eval_data = loadUserEvaluation(file_path)
        if eval_data:
            evaluations.append(eval_data)

    if not evaluations:
        return {"error": "No evaluation files found"}

    # Aggregate metrics
    all_ratings = []
    all_ndcg_scores = []
    input_methods = []

    for eval_data in evaluations:
        if 'ratings' in eval_data:
            all_ratings.extend(eval_data['ratings'].values())
        if 'metrics' in eval_data and 'ndcg' in eval_data['metrics']:
            all_ndcg_scores.append(eval_data['metrics']['ndcg'])
        if 'input_method' in eval_data:
            input_methods.append(eval_data['input_method'])

    return {
        "total_evaluations": len(evaluations),
        "average_rating": np.mean(all_ratings) if all_ratings else 0,
        "average_ndcg": np.mean(all_ndcg_scores) if all_ndcg_scores else 0,
        "input_method_distribution": {
            method: input_methods.count(method) for method in set(input_methods)
        },
        "rating_distribution": {
            f"rating_{i}": all_ratings.count(i) for i in range(1, 6)
        } if all_ratings else {}
    }

def generateEvaluationReport(ratings, recommendations, user_genres, recommended_genres, original_tags=None):
    """
    Generate a comprehensive evaluation report.

    Parameters:
    - ratings: Dictionary of game ratings
    - recommendations: DataFrame of recommended games
    - user_genres: List of user's preferred genres
    - recommended_genres: List of genres in recommendations
    - original_tags: Set of user's original tags (optional)

    Returns:
    - Dictionary with comprehensive metrics
    """
    relevance_scores = list(ratings.values())

    # Basic metrics
    ndcg = calculateNDCG(relevance_scores)
    avg_relevance = calculateRelevanceScore(relevance_scores)
    genre_coverage = calculateGenreCoverage(user_genres, recommended_genres)

    # Advanced metrics
    precision_at_5 = calculatePrecisionAtK(relevance_scores, k=5)
    recall_at_5 = calculateRecallAtK(relevance_scores, k=5)
    diversity = calculateDiversityScore(recommended_genres)
    satisfaction = calculateUserSatisfactionScore(relevance_scores)
    engagement = calculateEngagementMetrics(relevance_scores)

    report = {
        "basic_metrics": {
            "ndcg": ndcg,
            "average_relevance": avg_relevance,
            "genre_coverage": genre_coverage
        },
        "advanced_metrics": {
            "precision_at_5": precision_at_5,
            "recall_at_5": recall_at_5,
            "diversity_score": diversity,
            "user_satisfaction": satisfaction
        },
        "engagement_metrics": engagement,
        "system_info": {
            "total_recommendations": len(recommendations),
            "total_genres_recommended": len(set(recommended_genres)),
            "user_genres_count": len(user_genres),
            "original_tags_protected": len(original_tags) if original_tags else 0
        }
    }

    return report
