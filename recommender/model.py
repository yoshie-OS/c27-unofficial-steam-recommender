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

def createUserProfileFromSelection(selected_tags, unique_tags_df):
  """Create a user profile matrix from directly selected tags"""
  user_profile = pd.DataFrame({
      'tag': unique_tags_df['tag'],
      'tag_count': 0
  })

  for tag in selected_tags:
    if tag in user_profile['tag'].values:
      user_profile.loc[user_profile['tag'] == tag, 'tag_count'] = 1.0

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

def getRecommendations(normalized_user_profile, game_profiles, games_df, threshold=0.3, top_n=5):
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
