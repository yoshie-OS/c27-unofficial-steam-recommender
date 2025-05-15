import pandas as pd
import requests
from recommender import convertToSteamid64

def userDataCollector(steam_id, api_key=None) -> pd.DataFrame:
    """
    Get games owned by a specific user.
    Automatically converts vanity URLs or other Steam ID formats to SteamID64.

    Parameters:
    steam_id (str): The Steam ID of the user, can be SteamID64, custom URL, or other formats
    api_key (str): Your Steam API key

    Returns:
    DataFrame: Pandas DataFrame containing the user's games
    """
    # Check if steam_id is a valid SteamID64 (17-digit number)
    if not (str(steam_id).isdigit() and len(str(steam_id)) == 17):
        print(f"Input '{steam_id}' is not a SteamID64. Attempting to convert...")
        try:
            # Use the already implemented conversion function
            # Assuming the function is named convert_to_steamid64
            original_id = steam_id
            steam_id = convertToSteamid64(steam_id, api_key)

            if not steam_id:
                print(f"Failed to convert '{original_id}' to SteamID64.")
                return pd.DataFrame()

            print(f"Successfully converted to SteamID64: {steam_id}")
        except Exception as e:
            print(f"Error converting to SteamID64: {e}")
            return pd.DataFrame()

    # Now proceed with the valid SteamID64
    url = f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key={api_key}&steamid={steam_id}&include_appinfo=true&include_played_free_games=true"

    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()

            # Check if the response contains games
            if "games" not in data['response'] or not data['response']['games']:
                print(f"Tidak terdapat game pada user {steam_id} atau profil mungkin pribadi")
                return pd.DataFrame()

            # Extract the games data
            games = data['response']['games']

            # Process the games data directly (more robust approach)
            processed_games = []
            for game in games:
                game_data = {
                    "app_id": game.get("appid"),  # Steam API uses "appid" not "app_id"
                    "name": game.get("name", ""),
                    "playtime_forever": game.get("playtime_forever", 0),
                    "playtime_hours": round(game.get("playtime_forever", 0) / 60, 2),  # Convert minutes to hours
                    "playtime_2weeks": game.get("playtime_2weeks", 0),
                    "img_icon_url": game.get("img_icon_url", ""),
                    "img_logo_url": game.get("img_logo_url", "")
                }
                processed_games.append(game_data)

            # Create DataFrame from processed data
            library_df = pd.DataFrame(processed_games)

            # Print some statistics if needed
            total_games = len(library_df)
            if total_games > 0:
                total_playtime = library_df["playtime_forever"].sum()
                print(f"User {steam_id} owns {total_games} games")
                print(f"Total playtime: {total_playtime} minutes ({round(total_playtime/60, 2)} hours)")

            # Select and order columns to match user_library
            desired_columns = ['app_id', 'name', 'playtime_forever', 'playtime_2weeks',
                              'playtime_hours', 'img_icon_url', 'img_logo_url']

            # Only include columns that exist in our DataFrame
            available_columns = [col for col in desired_columns if col in library_df.columns]

            return library_df[available_columns]

        else:
            print(f"Gagal mengambil data user. Status code: {response.status_code}")
            if response.status_code == 400:
                print("Bad Request: Check if the SteamID64 is valid.")
            elif response.status_code == 401:
                print("Unauthorized: Check your API key.")
            elif response.status_code == 403:
                print("Forbidden: Your API key doesn't have the required permissions.")
            elif response.status_code == 429:
                print("Too Many Requests: You're being rate limited by the Steam API.")
            return pd.DataFrame()

    except Exception as e:
        print(f"Gagal mengambil data user: {e}")
        return pd.DataFrame()
