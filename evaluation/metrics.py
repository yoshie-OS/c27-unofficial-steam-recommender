import numpy as np
from pathlib import Path

def calculateNDCG(relevance_scores, k=10):
    if not relevance_scores:
        return 0.0

    # Calculate DCG (Discounted Cumulative Gain)
    dcg = relevance_scores[0]
    for i in range(1, min(k, len(relevance_scores))):
        dcg += relevance_scores[i] / np.log2(i + 2)

    # Calculate IDCG (Ideal DCG - relevance scores sorted in descending order)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = ideal_scores[0]
    for i in range(1, min(k, len(ideal_scores))):
        idcg += ideal_scores[i] / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg

def calculateRelevanceScore(ratings):
    if not ratings:
        return 0.0

    return sum(ratings) / len(ratings)

def calculateGenreCoverage(user_interests, recommended_genres):
    if not user_interests:
        return 0.0

    user_interests_set = set(user_interests)
    recommended_genres_set = set(recommended_genres)

    covered_genres = user_interests_set.intersection(recommended_genres_set)

    return (len(covered_genres) / len(user_interests_set)) * 100

def updateUserProfileWeights(user_profile, games_df, ratings, unique_tags_df):
    """Update user profile weights with EXTREMELY amplified effects"""
    # Create a copy of the user profile
    updated_profile = user_profile.copy()

    # Track which tags to boost and which to penalize
    tags_to_boost = set()
    tags_to_penalize = set()

    # First, identify tags to boost or penalize based on ratings
    for app_id, rating in ratings.items():
        # Get the game's tags
        game_row = games_df[games_df['id'] == int(app_id)]
        if not game_row.empty:
            # Extract tags from the game
            game_tags_str = game_row['tags'].iloc[0]
            # Convert string representation to list
            if isinstance(game_tags_str, str):
                game_tags_str = game_tags_str.replace('[', '').replace(']', '').replace('"', '')
                game_tags = [tag.strip() for tag in game_tags_str.split(',')]

                # Determine whether to boost or penalize
                if rating >= 4:  # High rating
                    tags_to_boost.update(game_tags)
                elif rating <= 2:  # Low rating
                    tags_to_penalize.update(game_tags)

    # Remove overlaps (don't penalize tags that should also be boosted)
    tags_to_penalize = tags_to_penalize - tags_to_boost

    # Apply EXTREME modifications - multiply original weights
    for idx, row in updated_profile.iterrows():
        tag = row['tag']
        original_weight = row['tag_count']

        if tag in tags_to_boost:
            # BOOST: Triple the weight for liked tags
            updated_profile.at[idx, 'tag_count'] = original_weight * 3.0
        elif tag in tags_to_penalize:
            # PENALIZE: Reduce to almost zero for disliked tags
            updated_profile.at[idx, 'tag_count'] = original_weight * 0.1  # 90% reduction

        # Ensure weight doesn't go negative
        if updated_profile.at[idx, 'tag_count'] < 0:
            updated_profile.at[idx, 'tag_count'] = 0

    # Identify the top 5 tags from highly rated games to further emphasize
    top_rated_tags = list(tags_to_boost)[:5]

    # Find "cousin" tags that often appear with top rated tags
    cousin_tags = set()
    for _, game in games_df.iterrows():
        game_tags_str = game['tags']
        if isinstance(game_tags_str, str):
            game_tags_str = game_tags_str.replace('[', '').replace(']', '').replace('"', '')
            game_tags = [tag.strip() for tag in game_tags_str.split(',')]

            # Check if the game has any top rated tags
            if any(tag in top_rated_tags for tag in game_tags):
                # Add other tags from this game as "cousins"
                for tag in game_tags:
                    if tag not in top_rated_tags and tag not in tags_to_penalize:
                        cousin_tags.add(tag)

    # Give a boost to "cousin" tags to increase diversity
    for idx, row in updated_profile.iterrows():
        tag = row['tag']
        if tag in cousin_tags:
            # Boost cousin tags (but less than direct boosts)
            updated_profile.at[idx, 'tag_count'] *= 1.5

    return updated_profile

def saveUserEvaluation(user_id, input_method, selected_tags, recommendations, ratings, metrics):
    data_dir = Path("user_evaluations")
    data_dir.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if user_id:
        filename = f"user_{user_id}_{timestamp}.json"
    else:
        filename = f"anonymous_user_{timestamp}.json"

    data = {
    "timestamp": timestamp,
    "user_id": user_id,
    "input_method": input_method,
    "selected_tags": selected_tags,
    "recommendations": recommendations.to_dict('records') if isinstance(recommendations, pd.DataFrame) else recommendations,
    "ratings": ratings,
    "metrics": metrics
    }

    file_path = data_dir / filename
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    return file_path

def collectUserFeedback(recommendations, user_consent=False):
    ratings = {}
    for _, game in recommendations.iterrows():
        ratings[game['app_id']] = 3

    return ratings

def extractUserGenres(user_profile):
    user_genres = user_profile[user_profile['tag_count'] > 0]['tag'].tolist()
    return user_genres

def extractRecommendedGenres(recommendations, games_df):
    all_genres = []

    for _, game in recommendations.iterrows():
        game_row = games_df[games_df['id'] == game['app_id']]

        if not game_row.empty:
            game_tags_str = game_row['tags'].iloc[0]

            if isinstance(game_tags_str, str):
                game_tags_str = game_tags_str.replace('[', '').replace(']', '').replace('"', '')
                game_tags = [tag.strip() for tag in game_tags_str.split(',')]
                all_genres.extend(game_tags)
    return list(set(all_genres))
