import pandas as pd
import requests
import re
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def create_robust_session():
    """
    Create a requests session with retry strategy for better reliability.
    This helps handle temporary network issues and rate limiting.
    """
    session = requests.Session()

    # Define retry strategy
    retry_strategy = Retry(
        total=3,  # Total number of retries
        backoff_factor=1,  # Wait time between retries (1, 2, 4 seconds)
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
        allowed_methods=["HEAD", "GET", "OPTIONS"]  # Methods to retry (updated from method_whitelist)
    )

    # Mount adapter with retry strategy
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

def convert_to_steamid64(identifier, api_key):
    """
    Enhanced Steam ID conversion with better error handling and validation.
    """
    if not identifier or not api_key:
        return None

    identifier = str(identifier).strip()

    # Validate identifier isn't empty after stripping
    if not identifier:
        return None

    # Already a SteamID64 (17-digit number starting with 7656119)
    if re.match(r'^7656119\d{10}$', identifier):
        return identifier

    # SteamID format (STEAM_0:X:XXXXXXXX or STEAM_1:X:XXXXXXXX)
    steamid_match = re.match(r'^STEAM_[01]:([01]):(\d+)$', identifier)
    if steamid_match:
        y = int(steamid_match.group(1))
        z = int(steamid_match.group(2))
        steamid64 = str(76561197960265728 + y + (z * 2))
        return steamid64

    # SteamID3 format ([U:1:XXXXXXXX])
    steamid3_match = re.match(r'^\[U:1:(\d+)\]$', identifier)
    if steamid3_match:
        account_id = int(steamid3_match.group(1))
        return str(76561197960265728 + account_id)

    # Handle full Steam profile URLs
    url_patterns = [
        r'steamcommunity\.com/profiles/(\d{17})',
        r'steamcommunity\.com/id/([^/]+)',
    ]

    for pattern in url_patterns:
        match = re.search(pattern, identifier)
        if match:
            extracted = match.group(1)
            # If it's already a SteamID64 from URL
            if re.match(r'^7656119\d{10}$', extracted):
                return extracted
            else:
                # It's a custom URL, continue to resolve below
                identifier = extracted
                break

    # Try to resolve as custom URL using Steam API
    session = create_robust_session()
    resolve_url = f"https://api.steampowered.com/ISteamUser/ResolveVanityURL/v1/"

    try:
        params = {
            'key': api_key,
            'vanityurl': identifier,
            'url_type': 1  # Individual profile
        }

        response = session.get(resolve_url, params=params, timeout=10)
        response.raise_for_status()  # Raises an HTTPError for bad responses

        data = response.json()

        if data.get('response', {}).get('success') == 1:
            return data['response']['steamid']
        elif data.get('response', {}).get('success') == 42:
            print(f"Steam ID '{identifier}' not found")
            return None
        else:
            print(f"Failed to resolve vanity URL: {data.get('response', {})}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Network error resolving vanity URL: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error resolving vanity URL: {e}")
        return None
    finally:
        session.close()

def userDataCollector(steam_id, api_key=None) -> pd.DataFrame:
    """
    Enhanced Steam library collector with comprehensive error handling.
    """
    if not steam_id or not api_key:
        print("Error: Steam ID and API key are required")
        return pd.DataFrame()

    # Convert to SteamID64 if necessary
    original_id = steam_id
    if not (str(steam_id).isdigit() and len(str(steam_id)) == 17):
        print(f"Converting '{steam_id}' to SteamID64...")
        steam_id = convert_to_steamid64(steam_id, api_key)

        if not steam_id:
            print(f"Failed to convert '{original_id}' to valid SteamID64")
            return pd.DataFrame()

        print(f"Successfully converted to SteamID64: {steam_id}")

    # Create robust session for API calls
    session = create_robust_session()

    # Steam API endpoint for owned games
    url = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"
    params = {
        'key': api_key,
        'steamid': steam_id,
        'include_appinfo': 'true',
        'include_played_free_games': 'true',
        'format': 'json'
    }

    try:
        print(f"Fetching games for Steam ID: {steam_id}")
        response = session.get(url, params=params, timeout=15)

        # Handle specific HTTP status codes
        if response.status_code == 401:
            print("Error 401: Invalid API key")
            return pd.DataFrame()
        elif response.status_code == 403:
            print("Error 403: API key doesn't have required permissions")
            return pd.DataFrame()
        elif response.status_code == 429:
            print("Error 429: Rate limited by Steam API. Please wait and try again.")
            return pd.DataFrame()
        elif response.status_code == 500:
            print("Error 500: Steam API server error. Please try again later.")
            return pd.DataFrame()

        response.raise_for_status()  # Raise exception for other HTTP errors

        data = response.json()

        # Validate response structure
        if 'response' not in data:
            print("Invalid response format from Steam API")
            return pd.DataFrame()

        response_data = data['response']

        # Check if profile is private or has no games
        if 'games' not in response_data:
            print(f"No games found for user {steam_id}. Profile might be private or user has no games.")
            return pd.DataFrame()

        games = response_data['games']

        if not games:
            print(f"User {steam_id} has no games in their library")
            return pd.DataFrame()

        # Process games data with validation
        processed_games = []
        for game in games:
            try:
                # Validate required fields
                if 'appid' not in game:
                    continue

                game_data = {
                    "app_id": int(game["appid"]),  # Ensure integer type
                    "name": str(game.get("name", "Unknown Game")),
                    "playtime_forever": int(game.get("playtime_forever", 0)),
                    "playtime_2weeks": int(game.get("playtime_2weeks", 0)),
                    "img_icon_url": str(game.get("img_icon_url", "")),
                    "img_logo_url": str(game.get("img_logo_url", ""))
                }

                # Calculate playtime in hours
                game_data["playtime_hours"] = round(game_data["playtime_forever"] / 60, 2)

                processed_games.append(game_data)

            except (ValueError, TypeError) as e:
                print(f"Error processing game data: {e}")
                continue

        if not processed_games:
            print("No valid games found after processing")
            return pd.DataFrame()

        # Create DataFrame
        library_df = pd.DataFrame(processed_games)

        # Validate DataFrame structure
        required_columns = ['app_id', 'name', 'playtime_forever']
        missing_columns = [col for col in required_columns if col not in library_df.columns]

        if missing_columns:
            print(f"Warning: Missing required columns: {missing_columns}")
            return pd.DataFrame()

        # Print statistics
        total_games = len(library_df)
        total_playtime = library_df["playtime_forever"].sum()
        print(f"Successfully retrieved {total_games} games for user {steam_id}")
        print(f"Total playtime: {total_playtime} minutes ({round(total_playtime/60, 2)} hours)")

        return library_df

    except requests.exceptions.Timeout:
        print("Request timed out. Steam API might be slow.")
        return pd.DataFrame()
    except requests.exceptions.ConnectionError:
        print("Connection error. Check your internet connection.")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching user data: {e}")
        return pd.DataFrame()
    except ValueError as e:
        print(f"Error parsing JSON response: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error fetching user data: {e}")
        return pd.DataFrame()
    finally:
        session.close()
