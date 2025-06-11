import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

def createUserProfile(user_library, tags_df, unique_tags_df):
    """Create user profile in the form of matrix"""
    # Initialize a DataFrame with tags and zero counts
    user_profile = pd.DataFrame({
        'tag': unique_tags_df['tag'],
        'tag_count': [0] * len(unique_tags_df)
    })

    # Check if user_library is empty (API error or no games)
    if user_library.empty:
        return user_profile

    # Extract the app_ids from the user's library
    user_app_ids = user_library['app_id'].tolist()

    # Filter tags_df to only include games in the user's library
    user_tags = tags_df[tags_df['app_id'].isin(user_app_ids)]

    # Count the frequency of each tag in the user's library
    tag_counts = user_tags['tag'].value_counts().reset_index()
    tag_counts.columns = ['tag', 'tag_count']

    # Update the user profile with the actual counts
    user_profile = user_profile.merge(tag_counts, on='tag', how='left', suffixes=('', '_actual'))
    user_profile['tag_count'] = user_profile['tag_count_actual'].fillna(0)
    user_profile = user_profile[['tag', 'tag_count']]

    return user_profile

def createUserProfileFromSelection(tag_weights, unique_tags_df):
    """
    Create a user profile matrix from selected tags with importance weights.

    Parameters:
    - tag_weights: Dictionary mapping tag names to importance weights (1-5)
    - unique_tags_df: DataFrame containing all unique tags

    Returns:
    - DataFrame with user profile
    """
    user_profile = pd.DataFrame({
        'tag': unique_tags_df['tag'],
        'tag_count': 0.0
    })

    # Apply weights to selected tags
    for tag, weight in tag_weights.items():
        if tag in user_profile['tag'].values:
            user_profile.loc[user_profile['tag'] == tag, 'tag_count'] = float(weight)

    return user_profile

def normalizeUserProfile(user_profile):
    """Normalizes the user profile"""
    # Extract the tag counts as a numpy array
    tag_counts = user_profile['tag_count'].values.reshape(1, -1)

    # Apply L2 normalization
    normalized_counts = normalize(tag_counts, norm='l2')[0]

    # Update the user profile with normalized counts
    normalized_profile = user_profile.copy()
    normalized_profile['tag_count'] = normalized_counts

    return normalized_profile

def createGameProfiles(tags_df, unique_tags_df, games_df):
    """Create game profiles (for all games in the database)"""
    # Create a pivot table: rows=games, columns=tags, values=binary(1/0) (has tag or not)
    game_tag_matrix = pd.crosstab(tags_df['app_id'], tags_df['tag'])

    # Ensure all tags are included
    for tag in unique_tags_df['tag']:
        if tag not in game_tag_matrix.columns:
            game_tag_matrix[tag] = 0

    # Order columns according to unique_tags_df to match user profile
    game_tag_matrix = game_tag_matrix[unique_tags_df['tag']]

    # Normalize each game's profile
    normalized_matrix = normalize(game_tag_matrix.values, norm='l2', axis=1)
    game_tag_matrix = pd.DataFrame(
        normalized_matrix,
        index=game_tag_matrix.index,
        columns=game_tag_matrix.columns
    )

    return game_tag_matrix

def getRecommendations(normalized_user_profile, game_profiles, games_df, user_library=None, threshold=0.3, top_n=5):
    """
    Get game recommendations based on user profile.

    Parameters:
    - normalized_user_profile: Normalized user profile DataFrame
    - game_profiles: Game profiles matrix
    - games_df: DataFrame containing game information
    - user_library: Optional DataFrame containing user's library (to exclude owned games)
    - threshold: Similarity threshold (default: 0.3)
    - top_n: Number of recommendations to return (default: 5)

    Returns:
    - DataFrame of recommended games
    """
    # Get user profile as a vector
    user_vector = normalized_user_profile['tag_count'].values

    # Calculate COSINE SIMILARITY between user profile and all games
    similarities = cosine_similarity([user_vector], game_profiles.values)[0]

    # Create a DataFrame with app_ids and similarity scores
    app_ids = game_profiles.index
    similarity_df = pd.DataFrame({
        'app_id': app_ids,
        'similarity': similarities
    })

    # Filter by threshold and sort
    recommendations = similarity_df[similarity_df['similarity'] >= threshold].sort_values('similarity', ascending=False)

    # Remove games that are already in user's library (if library exists and isn't empty)
    if user_library is not None and not user_library.empty and 'app_id' in user_library.columns:
        user_app_ids = set(user_library['app_id'].tolist())
        recommendations = recommendations[~recommendations['app_id'].isin(user_app_ids)]

    # Get top N recommendations
    top_recommendations = recommendations.head(top_n)

    # Join with game information to get game names
    recommended_games = top_recommendations.merge(games_df, left_on='app_id', right_on='id', how='left')

    return recommended_games[['app_id', 'name', 'similarity']]

def extract_game_tags(app_id, games_df):
    """Helper function to extract tags from a game"""
    game_row = games_df[games_df['id'] == int(app_id)]
    if not game_row.empty:
        game_tags_str = game_row['tags'].iloc[0]
        if isinstance(game_tags_str, str):
            game_tags_str = game_tags_str.replace('[', '').replace(']', '').replace('"', '').replace("'", "")
            return [tag.strip() for tag in game_tags_str.split(',') if tag.strip()]
    return []

def updateTagRelevancyWithResistance(ratings, games_df, current_tag_relevancy=None):
    """
    Enhanced tag relevancy with 3-tier resistance system.
    Rating 3 gets slight negative instead of neutral to provide resistance.
    """
    if current_tag_relevancy is None:
        tag_relevancy = {}
    else:
        tag_relevancy = current_tag_relevancy.copy()

    # Process each rated game
    for app_id, rating in ratings.items():
        game_tags = extract_game_tags(app_id, games_df)

        # Enhanced relevancy calculation with resistance
        if rating >= 4:
            relevancy_adjustment = (rating - 3) / 2.0  # +0.5 to +1.0
        elif rating <= 2:
            relevancy_adjustment = (rating - 3) / 2.0  # -0.5 to -1.0
        else:  # rating == 3
            relevancy_adjustment = -0.1  # Slight negative (medium resistance)

        # Update relevancy for each tag
        for tag in game_tags:
            if tag in tag_relevancy:
                # Average with existing score (to prevent wild swings)
                tag_relevancy[tag] = (tag_relevancy[tag] + relevancy_adjustment) / 2
            else:
                # New tag
                tag_relevancy[tag] = relevancy_adjustment

    return tag_relevancy

def updateUserProfileWith3TierResistance(user_profile, games_df, ratings, unique_tags_df, original_user_tags):
    """
    Update user profile with enhanced 3-tier resistance system and strong original tag protection
    
    Parameters:
    -----------
    user_profile : DataFrame
        User profile with tag counts
    games_df : DataFrame
        Game data containing tags
    ratings : dict
        Dictionary of game ratings from user feedback
    unique_tags_df : DataFrame
        DataFrame of all unique tags
    original_user_tags : set
        Set of original user tags to protect from elimination
        
    Returns:
    --------
    DataFrame
        Updated user profile with adjusted tag counts
    """
    # Create a copy to avoid modifying original
    updated_profile = user_profile.copy()
    
    # Store original tag values for protection
    original_values = {}
    for tag in original_user_tags:
        if tag in updated_profile['tag'].values:
            original_values[tag] = updated_profile.loc[updated_profile['tag'] == tag, 'tag_count'].values[0]
    
    # Process each rated game
    for app_id, rating in ratings.items():
        # Get game tags - handle different column names
        if 'id' in games_df.columns:
            game_row = games_df[games_df['id'] == app_id]
        else:
            game_row = games_df[games_df['app_id'] == app_id]
            
        if game_row.empty:
            continue
            
        game_tags = game_row['tags'].values[0].split(',')
        
        for tag in game_tags:
            # Skip if tag not in profile
            if tag not in updated_profile['tag'].values:
                continue
                
            # Calculate adjustment based on rating (3-tier system)
            if rating <= 2:  # Negative feedback (1-2 stars)
                adjustment = -0.15  # Penalty for disliked games
            elif rating == 3:  # Neutral feedback (3 stars)
                adjustment = 0.05   # Small boost for neutral games
            else:  # Positive feedback (4-5 stars)
                adjustment = 0.25   # Larger boost for liked games
            
            # Apply adjustment
            current_value = updated_profile.loc[updated_profile['tag'] == tag, 'tag_count'].values[0]
            new_value = max(0, current_value + adjustment)
            updated_profile.loc[updated_profile['tag'] == tag, 'tag_count'] = new_value
    
    # ENHANCED PROTECTION LOGIC: Protect original user interests
    for tag, original_value in original_values.items():
        current_value = updated_profile.loc[updated_profile['tag'] == tag, 'tag_count'].values[0]
        
        # Calculate minimum allowed value (30% of original or absolute minimum 0.15)
        min_allowed = max(original_value * 0.3, 0.15)
        
        # If current value is below minimum allowed, restore it
        if current_value < min_allowed:
            updated_profile.loc[updated_profile['tag'] == tag, 'tag_count'] = min_allowed
            
    # Final safety check for original tags
    for tag in original_user_tags:
        if tag in updated_profile['tag'].values:
            if updated_profile.loc[updated_profile['tag'] == tag, 'tag_count'].values[0] <= 0:
                # Force a minimum value for original tags to ensure they're never eliminated
                updated_profile.loc[updated_profile['tag'] == tag, 'tag_count'] = 0.15
    
    return updated_profile

def calculateDynamicThreshold(app_id, games_df, tag_relevancy, base_threshold):
    """
    Calculate dynamic threshold for a game based on its tags and tag relevancy scores.
    Enhanced with better error handling and type safety.
    """
    try:
        # Ensure consistent data types for comparison
        app_id_int = int(app_id)

        # Find the game in games_df - handle both 'id' and 'app_id' column names
        if 'id' in games_df.columns:
            game_row = games_df[games_df['id'] == app_id_int]
        elif 'app_id' in games_df.columns:
            game_row = games_df[games_df['app_id'] == app_id_int]
        else:
            print(f"Warning: No 'id' or 'app_id' column found in games_df")
            return base_threshold

        if game_row.empty:
            return base_threshold

        # Extract tags from the game with better error handling
        game_tags = extract_game_tags(app_id, games_df)

        if not game_tags:
            return base_threshold

        # Calculate average relevancy for this game's tags
        relevancy_scores = []
        for tag in game_tags:
            if tag in tag_relevancy and isinstance(tag_relevancy[tag], (int, float)):
                relevancy_scores.append(tag_relevancy[tag])

        if not relevancy_scores:
            return base_threshold

        avg_relevancy = sum(relevancy_scores) / len(relevancy_scores)

        # Adjust threshold based on average relevancy with bounds checking
        if avg_relevancy > 0.6:        # High relevancy - easier to recommend
            return max(base_threshold * 0.67, 0.1)  # Don't go below 0.1
        elif avg_relevancy > 0.3:      # Medium-high relevancy
            return base_threshold * 0.83
        elif avg_relevancy < -0.3:     # Low relevancy - much harder to recommend
            return min(base_threshold * 2.0, 0.9)   # Don't exceed 0.9
        elif avg_relevancy < 0:        # Medium-low relevancy
            return base_threshold * 1.33
        else:
            return base_threshold

    except Exception as e:
        print(f"Error in calculateDynamicThreshold for app_id {app_id}: {e}")
        return base_threshold

def calculateThresholdWithResistance(app_id, games_df, tag_relevancy, base_threshold, neutral_games=None):
    """
    Calculate threshold with resistance for neutral games.
    Rating 3 games get medium threshold (not too easy, not too hard to appear).
    """
    if neutral_games and app_id in neutral_games:
        # Medium threshold for rating 3 games (slightly harder than normal)
        return base_threshold * 1.2
    else:
        # Normal dynamic threshold logic
        return calculateDynamicThreshold(app_id, games_df, tag_relevancy, base_threshold)

def getRecommendationsWithNeutralProtection(normalized_user_profile, game_profiles, games_df, user_library=None,
                                           blacklisted_games=None, whitelisted_games=None, neutral_games=None,
                                           tag_relevancy=None, base_threshold=0.3, top_n=10):
    """
    Enhanced recommendation function with 3-tier protection system.

    Parameters:
    - normalized_user_profile: Normalized user profile DataFrame
    - game_profiles: Game profiles matrix
    - games_df: DataFrame containing game information
    - user_library: Optional DataFrame containing user's library (to exclude owned games)
    - blacklisted_games: Set of game IDs to completely exclude
    - whitelisted_games: Set of game IDs to prioritize
    - neutral_games: Set of game IDs with medium resistance
    - tag_relevancy: Dictionary mapping tags to relevancy scores
    - base_threshold: Base similarity threshold (default: 0.3)
    - top_n: Number of recommendations to return (default: 10)

    Returns:
    - DataFrame of recommended games
    """
    # Get user profile as a vector
    user_vector = normalized_user_profile['tag_count'].values

    # Calculate COSINE SIMILARITY between user profile and all games
    similarities = cosine_similarity([user_vector], game_profiles.values)[0]

    # Create a DataFrame with app_ids and similarity scores
    app_ids = game_profiles.index
    similarity_df = pd.DataFrame({
        'app_id': app_ids,
        'similarity': similarities
    })

    # Apply blacklist filtering (completely exclude blacklisted games)
    if blacklisted_games:
        similarity_df = similarity_df[~similarity_df['app_id'].isin(blacklisted_games)]

    # Remove games already in user's library
    if user_library is not None and not user_library.empty and 'app_id' in user_library.columns:
        user_app_ids = set(user_library['app_id'].tolist())
        similarity_df = similarity_df[~similarity_df['app_id'].isin(user_app_ids)]

    # Apply 3-tier dynamic thresholds
    filtered_recommendations = []

    for _, row in similarity_df.iterrows():
        app_id = row['app_id']
        similarity = row['similarity']

        # Determine threshold and boost based on tier
        if whitelisted_games and app_id in whitelisted_games:
            # Whitelist: very low threshold + boost
            threshold = base_threshold * 0.5
            similarity *= 1.3
            tier = 'whitelist'
        elif neutral_games and app_id in neutral_games:
            # Neutral: medium threshold (medium resistance)
            threshold = base_threshold * 1.2
            tier = 'neutral'
        else:
            # Unknown: dynamic threshold based on tags
            if tag_relevancy:
                threshold = calculateDynamicThreshold(app_id, games_df, tag_relevancy, base_threshold)
            else:
                threshold = base_threshold
            tier = 'discovery'

        # Include if similarity meets threshold
        if similarity >= threshold:
            filtered_recommendations.append({
                'app_id': app_id,
                'similarity': similarity,
                'tier': tier
            })

    recommendations = pd.DataFrame(filtered_recommendations)

    if recommendations.empty:
        return pd.DataFrame(columns=['app_id', 'name', 'similarity'])

    # Sort by similarity (descending)
    recommendations = recommendations.sort_values('similarity', ascending=False)

    # Get top N recommendations
    top_recommendations = recommendations.head(top_n)

    # Join with game information to get game names
    recommended_games = top_recommendations.merge(games_df, left_on='app_id', right_on='id', how='left')

    return recommended_games[['app_id', 'name', 'similarity']]

# Legacy functions for backward compatibility
def updateUserProfileWithEnhancedWeights(user_profile, games_df, ratings, unique_tags_df):
    """
    Legacy function - redirects to new 3-tier system
    """
    return updateUserProfileWith3TierResistance(user_profile, games_df, ratings, unique_tags_df)

def updateTagRelevancy(ratings, games_df, current_tag_relevancy=None):
    """
    Legacy function - redirects to new resistance system
    """
    return updateTagRelevancyWithResistance(ratings, games_df, current_tag_relevancy)

def getRecommendationsWithFiltering(normalized_user_profile, game_profiles, games_df, user_library=None,
                                   blacklisted_games=None, whitelisted_games=None, tag_relevancy=None,
                                   base_threshold=0.3, top_n=10):
    """
    Legacy function - redirects to new neutral protection system
    """
    return getRecommendationsWithNeutralProtection(
        normalized_user_profile, game_profiles, games_df, user_library,
        blacklisted_games, whitelisted_games, None, tag_relevancy,
        base_threshold, top_n
    )
