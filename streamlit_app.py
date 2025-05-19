import streamlit as st
import pandas as pd
from pathlib import Path

# Import your existing functions
from api.connector import userDataCollector
from api.utils import decrypt_api_key

# Import recommender functions
from recommender.model import (
    createUserProfile,
    createUserProfileFromSelection,
    normalizeUserProfile,
    createGameProfiles,
    getRecommendations
)

# Import evaluation functions (new package)
from evaluation.metrics import (
    calculateNDCG,
    calculateRelevanceScore,
    calculateGenreCoverage,
    updateUserProfileWeights,
    extractUserGenres,
    extractRecommendedGenres,
    saveUserEvaluation
)

# Page configuration
st.set_page_config(
    page_title="C27 Unofficial Steam Recommender",
    page_icon="🎮",
    layout="centered"
)

# Initialize session state for evaluation
if 'evaluation_submitted' not in st.session_state:
    st.session_state.evaluation_submitted = False

# Create directory for saving user evaluations
Path("user_evaluations").mkdir(exist_ok=True)

# App title and description
st.title("C27 Unofficial Steam Recommender")
st.write("Get personalized game recommendations based on your preferences.")

# Load data
@st.cache_data
def load_data():
    games_df = pd.read_csv('data/games_filtered.csv')
    tags_df = pd.read_csv('data/filtered_tags.csv')
    unique_tags_df = pd.DataFrame({'tag': tags_df['tag'].unique()})
    return games_df, tags_df, unique_tags_df

games_df, tags_df, unique_tags_df = load_data()

# Input method selection
option = st.radio("Choose your input method:", ["By Steam ID", "By Categories"])

if option == "By Steam ID":
    steam_id = st.text_input("Enter your Steam ID or custom URL:",
                           help="You can enter your Steam ID, custom URL, or SteamID64.")
    if st.button("Find Games by Steam ID", key="steam_id_button") and steam_id:
        with st.spinner("Fetching your games library..."):
            # Get API key
            api_key = decrypt_api_key()

            # Get user library
            user_library = userDataCollector(steam_id, api_key)

            # Check if user_library is empty
            if user_library.empty:
                st.warning("Could not retrieve your Steam library. Please check your Steam ID or make sure your profile is public.")
                st.info("Switching to category selection mode...")
                option = "By Categories"
            else:
                # Create user profile
                user_profile = createUserProfile(user_library, tags_df, unique_tags_df)
                normalized_user_profile = normalizeUserProfile(user_profile)

                # Create game profiles
                game_profiles = createGameProfiles(tags_df, unique_tags_df, games_df)

                # Get recommendations - explicitly pass the user_library
                recommendations = getRecommendations(
                    normalized_user_profile,
                    game_profiles,
                    games_df,
                    user_library=user_library,  # Pass the user library explicitly
                    threshold=0.3,
                    top_n=10
                )

                # Store in session state for later use
                st.session_state.recommendations = recommendations
                st.session_state.user_profile = user_profile
                st.session_state.normalized_user_profile = normalized_user_profile
                st.session_state.game_profiles = game_profiles
                st.session_state.user_id = steam_id
                st.session_state.input_method = "steam_id"
                st.session_state.selected_tags = None
                st.session_state.user_library = user_library  # Store user_library in session state

elif option == "By Categories":
    # Display categories for selection
    available_tags = sorted(unique_tags_df['tag'].tolist())
    selected_tags = st.multiselect("Select your preferred game genres:", available_tags)

    if st.button("Find Games by Categories", key="categories_button") and selected_tags:
        with st.spinner("Generating recommendations..."):
            # Create user profile from selected tags
            user_profile = createUserProfileFromSelection(selected_tags, unique_tags_df)
            normalized_user_profile = normalizeUserProfile(user_profile)

            # Create game profiles
            game_profiles = createGameProfiles(tags_df, unique_tags_df, games_df)

            # Get recommendations - user_library is None for category selection
            recommendations = getRecommendations(
                normalized_user_profile,
                game_profiles,
                games_df,
                user_library=None,  # No user library for category selection
                threshold=0.3,
                top_n=10
            )

            # Store in session state for later use
            st.session_state.recommendations = recommendations
            st.session_state.user_profile = user_profile
            st.session_state.normalized_user_profile = normalized_user_profile
            st.session_state.game_profiles = game_profiles
            st.session_state.user_id = None
            st.session_state.input_method = "category"
            st.session_state.selected_tags = selected_tags
            st.session_state.user_library = None  # No user library for category method

# Display recommendations if available
if 'recommendations' in st.session_state and not st.session_state.recommendations.empty:
    recommendations = st.session_state.recommendations

    st.header("Recommended Games")

    # Display the recommendations in a nice format
    for i, game in recommendations.iterrows():
        col1, col2 = st.columns([1, 3])

        with col1:
            # we could show an image here if available
            image_url = f"https://cdn.akamai.steamstatic.com/steam/apps/{game['app_id']}/header.jpg"
            try:
                st.image(image_url, width=150)
            except exception as e:
                st.image("https://via.placeholder.com/150x100?text=game", width=150)

        with col2:
            st.subheader(game['name'])
            st.write(f"Similarity Score: {game['similarity']:.4f}")

            # Link to Steam store
            steam_url = f"https://store.steampowered.com/app/{game['app_id']}"
            st.markdown(f"[View on Steam]({steam_url})")

    # Display evaluation section
    st.header("Evaluation")

    # Get user consent
    st.markdown("We do not collect any of your data without your permission.</br> The data that we collect are ratings data that you have provided,</br> this data is used to better engineer the model to generate more accurate results.")
    user_consent = st.checkbox("I agree to share my ratings data for research purposes")

    if not st.session_state.evaluation_submitted:
        satisfaction = st.radio("Are you satisfied with these recommendations?", ["Yes", "No"])

        if satisfaction == "Yes":
            if st.button("Submit Positive Feedback", key="positive_feedback"):
                # For users who are satisfied, record perfect scores
                ratings = {game['app_id']: 5 for _, game in recommendations.iterrows()}

                # Calculate metrics with perfect scores
                relevance_scores = list(ratings.values())

                # Get user's genres of interest
                if st.session_state.input_method == "steam_id":
                    user_genres = extractUserGenres(st.session_state.user_profile)
                else:
                    user_genres = st.session_state.selected_tags

                # Get recommended genres
                recommended_genres = extractRecommendedGenres(recommendations, games_df)

                # Calculate metrics
                ndcg = calculateNDCG(relevance_scores)
                avg_relevance = calculateRelevanceScore(relevance_scores)
                genre_coverage = calculateGenreCoverage(user_genres, recommended_genres)

                metrics = {
                    "ndcg": ndcg,
                    "relevance_score": avg_relevance,
                    "genre_coverage": genre_coverage
                }

                # Display metrics
                st.subheader("Evaluation Results")
                st.write(f"NDCG (Normalized Discounted Cumulative Gain): {ndcg:.4f}")
                st.write(f"Average Relevance Score: {avg_relevance:.2f}/5.0")
                st.write(f"Genre Coverage: {genre_coverage:.2f}%")

                # Save evaluation data if user consented
                if user_consent:
                    saveUserEvaluation(
                        st.session_state.user_id,
                        st.session_state.input_method,
                        st.session_state.selected_tags,
                        recommendations,
                        ratings,
                        metrics
                    )
                    st.success("Thank you for your feedback! Your data has been recorded.")
                else:
                    st.info("Thank you for your feedback! Your data was not saved as you didn't provide consent.")

                st.session_state.evaluation_submitted = True

        elif satisfaction == "No":
            st.subheader("Please rate the relevance of each recommendation (1-5):")

            # Show sliders for rating each recommendation
            ratings = {}
            for i, game in recommendations.iterrows():
                ratings[game['app_id']] = st.slider(
                    f"How relevant is '{game['name']}' to your preferences?",
                    1, 5, 3, key=f"rating_{game['app_id']}"
                )

            if st.button("Submit Evaluation", key="negative_feedback"):
                # Calculate evaluation metrics
                relevance_scores = list(ratings.values())

                # Get user's genres of interest
                if st.session_state.input_method == "steam_id":
                    user_genres = extractUserGenres(st.session_state.user_profile)
                else:
                    user_genres = st.session_state.selected_tags

                # Get recommended genres
                recommended_genres = extractRecommendedGenres(recommendations, games_df)

                # Calculate metrics
                ndcg = calculateNDCG(relevance_scores)
                avg_relevance = calculateRelevanceScore(relevance_scores)
                genre_coverage = calculateGenreCoverage(user_genres, recommended_genres)

                metrics = {
                    "ndcg": ndcg,
                    "relevance_score": avg_relevance,
                    "genre_coverage": genre_coverage
                }

                # Display metrics
                st.subheader("Evaluation Results")
                st.write(f"NDCG (Normalized Discounted Cumulative Gain): {ndcg:.4f}")
                st.write(f"Average Relevance Score: {avg_relevance:.2f}/5.0")
                st.write(f"Genre Coverage: {genre_coverage:.2f}%")

                # Update user profile based on feedback
                updated_profile = updateUserProfileWeights(
                    st.session_state.user_profile,
                    games_df,
                    ratings,
                    unique_tags_df
                )
                normalized_updated_profile = normalizeUserProfile(updated_profile)

                # Display updated profile info
                st.subheader("Your Profile Has Been Updated")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Top 5 genres before feedback:**")
                    top_original = st.session_state.normalized_user_profile.sort_values('tag_count', ascending=False).head(5)
                    for _, row in top_original.iterrows():
                        st.write(f"- {row['tag']}: {row['tag_count']:.4f}")

                with col2:
                    st.markdown("**Top 5 genres after feedback:**")
                    top_updated = normalized_updated_profile.sort_values('tag_count', ascending=False).head(5)
                    for _, row in top_updated.iterrows():
                        st.write(f"- {row['tag']}: {row['tag_count']:.4f}")

                # Get new recommendations with updated profile
                new_recommendations = getRecommendations(
                    normalized_updated_profile,
                    st.session_state.game_profiles,
                    games_df,
                    user_library=st.session_state.user_library,  # Pass the user library from session state
                    threshold=0.3,
                    top_n=10
                )

                # Display new recommendations
                st.subheader("New Recommendations Based on Your Feedback")

                for i, game in new_recommendations.iterrows():
                    col1, col2 = st.columns([1, 3])

                    with col1:
                        # we could show an image here if available
                        image_url = f"https://cdn.akamai.steamstatic.com/steam/apps/{game['app_id']}/header.jpg"
                        try:
                            st.image(image_url, width=150)
                        except exception as e:
                            st.image("https://via.placeholder.com/150x100?text=game", width=150)

                    with col2:
                        st.subheader(game['name'])
                        st.write(f"Similarity Score: {game['similarity']:.4f}")

                        # Link to Steam store
                        steam_url = f"https://store.steampowered.com/app/{game['app_id']}"
                        st.markdown(f"[View on Steam]({steam_url})")

                # Save evaluation data if user consented
                if user_consent:
                    saveUserEvaluation(
                        st.session_state.user_id,
                        st.session_state.input_method,
                        st.session_state.selected_tags,
                        recommendations,
                        ratings,
                        metrics
                    )
                    st.success("Thank you for your feedback! Your data has been recorded.")
                else:
                    st.info("Thank you for your feedback! Your data was not saved as you didn't provide consent.")

                st.session_state.evaluation_submitted = True

    # Add a reset button to start over
    if st.session_state.evaluation_submitted:
        if st.button("Start Over", key="reset_button"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# Footer
st.markdown("---")
st.caption("C27 Unofficial Steam Recommender © 2025")
