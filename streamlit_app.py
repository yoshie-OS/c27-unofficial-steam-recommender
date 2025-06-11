import streamlit as st
import pandas as pd
from pathlib import Path
import base64

# Import your existing functions
from api.connector import userDataCollector
from api.utils import decrypt_api_key

# Import recommender functions
from recommender.model import (
    createUserProfile,
    createUserProfileFromSelection,
    normalizeUserProfile,
    createGameProfiles,
    getRecommendations,
    getRecommendationsWithNeutralProtection,
    updateUserProfileWith3TierResistance,
    updateTagRelevancyWithResistance
)

# Import evaluation functions
from evaluation.metrics import (
    calculateNDCG,
    calculateRelevanceScore,
    calculateGenreCoverage,
    extractUserGenres,
    extractRecommendedGenres,
    saveUserEvaluation
)

# Function to add Steam-inspired styling
def add_steam_bg():
    """
    Add Steam-inspired background and CSS styling
    """
    steam_bg = """
    <style>
        .stApp {
            background-color: #1b2838;
            color: #c7d5e0;
        }
        .stButton>button {
            background-color: #66c0f4;
            color: #1b2838;
            font-weight: bold;
            border: none;
        }
        .stButton>button:hover {
            background-color: #1e90ff;
        }
        h1, h2, h3 {
            color: #66c0f4 !important;
        }
        .stForm {
            background-color: #2a475e;
            padding: 20px;
            border-radius: 5px;
        }
        .stMetric {
            background-color: #2a475e;
            padding: 10px;
            border-radius: 5px;
        }
        .css-1d391kg {
            background-color: #2a475e;
        }
        div.stSlider > div[data-baseweb] > div > div {
            background-color: #66c0f4;
        }
        div.stSlider > div[data-baseweb] > div > div > div {
            background-color: #1e90ff;
        }
    </style>
    """
    st.markdown(steam_bg, unsafe_allow_html=True)



# Page configuration
st.set_page_config(
    page_title="C27 Unofficial Steam Recommender",
    page_icon="🎮",
    layout="centered"
)
add_steam_bg()

# Enhanced 3-tier session state initialization with original tag protection
if 'evaluation_submitted' not in st.session_state:
    st.session_state.evaluation_submitted = False
if 'blacklisted_games' not in st.session_state:
    st.session_state.blacklisted_games = set()
if 'whitelisted_games' not in st.session_state:
    st.session_state.whitelisted_games = set()
if 'neutral_games' not in st.session_state:
    st.session_state.neutral_games = set()
if 'tag_relevancy' not in st.session_state:
    st.session_state.tag_relevancy = {}
if 'show_improved' not in st.session_state:
    st.session_state.show_improved = False
if 'original_user_tags' not in st.session_state:
    st.session_state.original_user_tags = set()  # Track user's original interests

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

                # Track original user tags (from Steam library)
                original_tags = set(user_profile[user_profile['tag_count'] > 0]['tag'].tolist())
                st.session_state.original_user_tags = original_tags

                # Create game profiles
                game_profiles = createGameProfiles(tags_df, unique_tags_df, games_df)

                # Get recommendations
                recommendations = getRecommendations(
                    normalized_user_profile,
                    game_profiles,
                    games_df,
                    user_library=user_library,
                    threshold=0.3,
                    top_n=10
                )

                # Store in session state
                st.session_state.recommendations = recommendations
                st.session_state.user_profile = user_profile
                st.session_state.normalized_user_profile = normalized_user_profile
                st.session_state.game_profiles = game_profiles
                st.session_state.user_id = steam_id
                st.session_state.input_method = "steam_id"
                st.session_state.selected_tags = None
                st.session_state.user_library = user_library

elif option == "By Categories":
    st.subheader("Select your preferred game genres:")

    # Get available tags
    available_tags = sorted(unique_tags_df['tag'].tolist())

    # Let user select categories first
    selected_tags = st.multiselect("Choose genres you're interested in:", available_tags)

    # If tags are selected, show importance sliders
    tag_weights = {}
    if selected_tags:
        st.subheader("Rate the importance of each selected genre:")

        for tag in selected_tags:
            # Create slider with descriptive labels
            weight = st.slider(
                f"{tag}",
                min_value=1,
                max_value=5,
                value=3,
                key=f"weight_{tag}",
                help=f"How important is {tag} to you?"
            )

            # Add custom labels
            col1, col2, col3 = st.columns([1, 8, 1])
            with col1:
                st.markdown("<p style='text-align: left; font-size: 12px; color: #888; margin-top: -15px;'>Low Importance</p>", unsafe_allow_html=True)
            with col3:
                st.markdown("<p style='text-align: right; font-size: 12px; color: #888; margin-top: -15px;'>Critical Importance</p>", unsafe_allow_html=True)

            tag_weights[tag] = weight

    # Show generate button only if tags are selected
    if selected_tags and st.button("Find Games by Categories", key="categories_button"):
        with st.spinner("Generating recommendations..."):
            # Create user profile from selected tags with weights
            user_profile = createUserProfileFromSelection(tag_weights, unique_tags_df)
            normalized_user_profile = normalizeUserProfile(user_profile)

            # Track original user tags (from category selection)
            st.session_state.original_user_tags = set(selected_tags)

            # Create game profiles
            game_profiles = createGameProfiles(tags_df, unique_tags_df, games_df)

            # Get recommendations
            recommendations = getRecommendations(
                normalized_user_profile,
                game_profiles,
                games_df,
                user_library=None,
                threshold=0.3,
                top_n=10
            )

            # Store in session state
            st.session_state.recommendations = recommendations
            st.session_state.user_profile = user_profile
            st.session_state.normalized_user_profile = normalized_user_profile
            st.session_state.game_profiles = game_profiles
            st.session_state.user_id = None
            st.session_state.input_method = "category"
            st.session_state.selected_tags = selected_tags
            st.session_state.tag_weights = tag_weights
            st.session_state.user_library = None

# Display recommendations if available
if 'recommendations' in st.session_state and not st.session_state.recommendations.empty:
    recommendations = st.session_state.recommendations

    # EVALUATION RESULTS PAGE
    if st.session_state.evaluation_submitted:
        st.header("Evaluation Results")

        # Get stored data
        ratings = st.session_state.evaluation_ratings

        # ENHANCED 3-TIER EVALUATION LOGIC
        # 1. Update blacklist/whitelist/neutral based on ratings
        for app_id, rating in ratings.items():
            if rating <= 2:  # Low ratings go to blacklist
                st.session_state.blacklisted_games.add(app_id)
                # Remove from other lists if exists
                st.session_state.neutral_games.discard(app_id)
                st.session_state.whitelisted_games.discard(app_id)
            elif rating == 3:  # Neutral ratings go to safe list
                st.session_state.neutral_games.add(app_id)
                # Remove from other lists if exists
                st.session_state.blacklisted_games.discard(app_id)
                st.session_state.whitelisted_games.discard(app_id)
            elif rating >= 4:  # High ratings go to whitelist
                st.session_state.whitelisted_games.add(app_id)
                # Remove from other lists if exists
                st.session_state.blacklisted_games.discard(app_id)
                st.session_state.neutral_games.discard(app_id)

        # 2. Update tag relevancy scores with 3-tier system
        st.session_state.tag_relevancy = updateTagRelevancyWithResistance(
            ratings,
            games_df,
            st.session_state.tag_relevancy
        )

        # 3. Update user profile with 3-tier resistance and original tag protection
        updated_profile = updateUserProfileWith3TierResistance(
            st.session_state.user_profile,
            games_df,
            ratings,
            unique_tags_df,
            st.session_state.original_user_tags  # Pass original tags for protection
        )
        normalized_updated_profile = normalizeUserProfile(updated_profile)

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

        # Display metrics
        st.subheader("Your Ratings Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("NDCG Score", f"{ndcg:.3f}")
        with col2:
            st.metric("Avg Rating", f"{avg_relevance:.1f}/5.0")
        with col3:
            st.metric("Genre Coverage", f"{genre_coverage:.0f}%")

        # Display enhanced session tracking info
        if st.session_state.blacklisted_games or st.session_state.whitelisted_games or st.session_state.neutral_games:
            st.subheader("Session Learning Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"❌ Blacklisted: {len(st.session_state.blacklisted_games)}")
                st.caption("Games you disliked (won't show again)")
            with col2:
                st.write(f"👍 Neutral/Safe: {len(st.session_state.neutral_games)}")
                st.caption("Games you found decent (medium resistance)")
            with col3:
                st.write(f"⭐ Whitelisted: {len(st.session_state.whitelisted_games)}")
                st.caption("Games you loved (prioritized)")

        # Display updated profile info
        st.subheader("Your Updated Profile")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 5 genres before feedback:**")
            top_original = st.session_state.normalized_user_profile.sort_values('tag_count', ascending=False).head(5)
            for _, row in top_original.iterrows():
                protected_marker = "🔒" if row['tag'] in st.session_state.original_user_tags else ""
                st.write(f"- {row['tag']}: {row['tag_count']:.3f} {protected_marker}")

        with col2:
            st.markdown("**Top 5 genres after feedback:**")
            top_updated = normalized_updated_profile.sort_values('tag_count', ascending=False).head(5)
            for _, row in top_updated.iterrows():
                protected_marker = "🔒" if row['tag'] in st.session_state.original_user_tags else ""
                st.write(f"- {row['tag']}: {row['tag_count']:.3f} {protected_marker}")

        if st.session_state.original_user_tags:
            st.caption("🔒 = Protected original interests (can't go to zero)")
            
        # Debug information in an expandable section
        with st.expander("Debug Information (Tag Protection)"):
            st.write("**Protected Original Tags:**")
            
            # Get original tag values from initial profile
            original_profile = st.session_state.user_profile
            protected_tags_df = original_profile[original_profile['tag'].isin(st.session_state.original_user_tags)]
            
            # Create a comparison DataFrame
            debug_df = pd.DataFrame({
                'tag': protected_tags_df['tag'],
                'original_value': protected_tags_df['tag_count'],
                'updated_value': [updated_profile.loc[updated_profile['tag'] == tag, 'tag_count'].values[0] 
                                if tag in updated_profile['tag'].values else 0 
                                for tag in protected_tags_df['tag']]
            })
            
            # Calculate protection effectiveness
            debug_df['protected'] = debug_df['updated_value'] > 0
            debug_df['protection_level'] = (debug_df['updated_value'] / debug_df['original_value'] * 100).fillna(0).round(1)
            
            # Display the debug table
            st.dataframe(debug_df)
            
            # Display protection stats
            st.write(f"**Protection Status:** {debug_df['protected'].sum()}/{len(debug_df)} tags protected")
            avg_protection = debug_df['protection_level'].mean()
            st.write(f"**Average Protection Level:** {avg_protection:.1f}% of original value")

        # Store updated profile
        st.session_state.user_profile = updated_profile
        st.session_state.normalized_user_profile = normalized_updated_profile

        # Buttons for actions
        st.header("What's Next?")
        col1, col2, col3 = st.columns([2, 1, 2])

        with col1:
            if st.button("Get Improved Recommendations", key="get_improved_button"):
                st.session_state.show_improved = True

        with col2:
            st.markdown("<p style='text-align: center; margin: 20px 0; color: #666;'>or</p>", unsafe_allow_html=True)

        with col3:
            if st.button("Start Over", key="start_over_eval"):
                # Clear all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        # IMPROVED RECOMMENDATIONS SECTION
        if st.session_state.show_improved:
            st.markdown("---")
            st.header("Improved Recommendations")

            # Generate enhanced improved recommendations with 3-tier protection
            improved_recs = getRecommendationsWithNeutralProtection(
                normalized_updated_profile,
                st.session_state.game_profiles,
                games_df,
                user_library=st.session_state.user_library,
                blacklisted_games=st.session_state.blacklisted_games,
                whitelisted_games=st.session_state.whitelisted_games,
                neutral_games=st.session_state.neutral_games,
                tag_relevancy=st.session_state.tag_relevancy,
                base_threshold=0.2,
                top_n=10
            )

            if improved_recs.empty:
                st.warning("No improved recommendations found. Try adjusting your ratings or starting over.")
                st.write("**Debug Information:**")
                st.write(f"- Blacklisted games: {len(st.session_state.blacklisted_games)}")
                st.write(f"- Neutral games: {len(st.session_state.neutral_games)}")
                st.write(f"- Whitelisted games: {len(st.session_state.whitelisted_games)}")
                st.write(f"- Protected original tags: {st.session_state.original_user_tags}")
            else:
                improved_recs['app_id'] = improved_recs['app_id'].astype(int)

                # FORM untuk prevent auto-rerun setiap slider digeser
                with st.form("improved_ratings_form"):
                    st.write("Rate these improved recommendations:")

                    ratings_dict = {}

                    for i, game in improved_recs.iterrows():
                        col1, col2 = st.columns([1, 3])

                        with col1:
                            # Game image
                            image_url = f"https://cdn.akamai.steamstatic.com/steam/apps/{game['app_id']}/header.jpg"
                            st.image(image_url, width=150)

                        with col2:
                            st.subheader(game['name'])
                            st.write(f"Similarity Score: {game['similarity']:.4f}")

                            # Show tier information
                            if game['app_id'] in st.session_state.whitelisted_games:
                                st.write("⭐ *Prioritized - You loved similar games*")
                            elif game['app_id'] in st.session_state.neutral_games:
                                st.write("👍 *Safe choice - You found similar games decent*")
                            elif game['app_id'] in st.session_state.blacklisted_games:
                                st.write("❌ *This shouldn't appear (debug)*")
                            else:
                                st.write("🔍 *New discovery*")

                            # Steam link
                            steam_url = f"https://store.steampowered.com/app/{game['app_id']}"
                            st.markdown(f"[View on Steam]({steam_url})")

                            # Rating slider - INI GAK BAKAL AUTO-RERUN
                            rating = st.slider(
                                f"Rate {game['name']}",
                                min_value=1,
                                max_value=5,
                                value=3,
                                key=f"form_improved_{game['app_id']}",
                                help=f"How relevant is {game['name']} to your preferences?"
                            )

                            # Store in temporary dict
                            ratings_dict[game['app_id']] = rating

                        st.markdown("---")

                    # Submit button - BARU RUNNING SETELAH KLIK INI
                    submitted = st.form_submit_button("Evaluate These Improved Recommendations")

                    if submitted:
                        # Update evaluation ratings
                        all_ratings = st.session_state.evaluation_ratings.copy()
                        all_ratings.update(ratings_dict)

                        st.session_state.evaluation_ratings = all_ratings

                        # Reset improved state for next iteration
                        st.session_state.show_improved = False

                        # Rerun to show updated results
                        st.rerun()

    else:
        # ORIGINAL RECOMMENDATIONS PAGE
        st.header("Recommended Games")

        # Initialize ratings in session state
        if 'game_ratings' not in st.session_state:
            st.session_state.game_ratings = {}

        # Display recommendations with enhanced information
        for i, game in recommendations.iterrows():
            with st.container():
                st.markdown(f"""
                <div style="background-color: #2a475e; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 3])

                with col1:
                    # Game image
                    image_url = f"https://cdn.akamai.steamstatic.com/steam/apps/{game['app_id']}/header.jpg"
                    st.image(image_url, width=150)

                with col2:
                    st.subheader(game['name'])
                    
                    # Display similarity with visual indicator
                    similarity_pct = int(game['similarity'] * 100)
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <div style="width: 120px; font-size: 14px;">Match Score:</div>
                        <div style="flex-grow: 1; background-color: #14212b; height: 14px; border-radius: 7px; overflow: hidden;">
                            <div style="width: {similarity_pct}%; background-color: #66c0f4; height: 100%;"></div>
                        </div>
                        <div style="width: 60px; text-align: right; margin-left: 10px;">{similarity_pct}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Game tags (top 5)
                    # Handle different column names (either 'app_id' or 'id')
                    if 'id' in games_df.columns:
                        game_row = games_df[games_df['id'] == game['app_id']]
                    else:
                        game_row = games_df[games_df['app_id'] == game['app_id']]
                    
                    if not game_row.empty:
                        game_tags = game_row['tags'].values[0].split(',')[:5]
                        st.markdown('<div style="display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 10px;">' + 
                                    ''.join([f'<span style="background-color: #14212b; padding: 3px 8px; border-radius: 10px; font-size: 12px;">{tag}</span>' for tag in game_tags]) + 
                                    '</div>', unsafe_allow_html=True)

                    # Link to Steam store
                    steam_url = f"https://store.steampowered.com/app/{game['app_id']}"
                    st.markdown(f"<a href='{steam_url}' target='_blank' style='color: #66c0f4; text-decoration: none;'><div style='display: inline-flex; align-items: center; background-color: #14212b; padding: 5px 10px; border-radius: 3px; margin-top: 5px;'>🎮 View on Store Page</div></a>", unsafe_allow_html=True)

                    # Rating slider for each game
                    rating = st.slider(
                        f"Rate this game",
                        min_value=1,
                        max_value=5,
                        value=3,
                        key=f"game_rating_{game['app_id']}",
                        help=f"How relevant is {game['name']} to your preferences?"
                    )

                    # Add custom labels for the rating slider
                    col_left, col_middle, col_right = st.columns([3, 4, 3])
                    with col_left:
                        st.markdown("<p style='text-align: left; font-size: 12px; color: #888; margin-top: -15px; white-space: nowrap;'>Not Relevant</p>", unsafe_allow_html=True)
                    with col_right:
                        st.markdown("<p style='text-align: right; font-size: 12px; color: #888; margin-top: -15px; white-space: nowrap;'>Very Relevant</p>", unsafe_allow_html=True)

                    # Store rating in session state
                    st.session_state.game_ratings[game['app_id']] = rating

                # End of container div
                st.markdown("</div>", unsafe_allow_html=True)

        # Evaluation buttons section
        st.header("Submit Your Feedback")

        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            if st.button("Evaluate", key="evaluate_button"):
                # Use current slider ratings
                ratings = st.session_state.game_ratings.copy()

                # Store evaluation data
                st.session_state.evaluation_ratings = ratings
                st.session_state.evaluation_submitted = True
                st.rerun()

        with col2:
            st.markdown("<p style='text-align: center; margin: 20px 0; color: #666; font-size: 16px;'>or</p>", unsafe_allow_html=True)

        with col3:
            if st.button("Start Over", key="start_over_main"):
                # Clear all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

# Add project background information
if not any(key in st.session_state for key in ['recommendations', 'evaluation_submitted', 'show_improved']):
    st.markdown("---")
    with st.expander("About This Project", expanded=True):
        st.markdown("""
        ### 🎮 Steam Game Recommendation System
        
        This app provides **personalized Steam game recommendations** using content-based filtering with cosine similarity and user feedback learning.
        
        #### 🔍 How It Works:
        
        1. **Input Methods**:
           - **Steam ID**: Analyzes your game library automatically
           - **Category Selection**: Manually select genres you enjoy
           
        2. **Recommendation Technology**:
           - Creates a profile of your gaming preferences
           - Normalizes game and user vectors
           - Finds games similar to your profile using cosine similarity
           
        3. **Learning System**:
           - Uses a 3-tier feedback mechanism (Blacklisted, Neutral, Whitelisted)
           - Protects your core interests even after multiple feedback cycles
           - Improves recommendations based on your ratings
           
        #### 📊 Recommendation Quality:
        
        The system evaluates recommendations using metrics including:
        - **NDCG Score**: Measures ranking quality
        - **Average Rating**: Based on your feedback
        - **Genre Coverage**: How well recommendations match your interests

        #### 🔒 Privacy Note:
        No user data is stored permanently. Data is accessed only during your session and is not retained after you close the app.
        
        #### 🧪 Project Context:
        This is a capstone project demonstrating practical application of recommendation systems, machine learning concepts, and interactive user feedback.
        """)

# Footer
st.markdown("---")
st.caption("C27 Unofficial Steam Recommender © 2025 | Aplikasi dibuat menggunakan LLM Claude oleh Anthropic. Segala kebenaran masih perlu diverifikasi oleh ahli.")
