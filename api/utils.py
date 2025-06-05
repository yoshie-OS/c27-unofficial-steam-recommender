import streamlit as st

def decrypt_api_key():
    """
    Get Steam API key from Streamlit secrets.
    Originally used encryption but now uses Streamlit's secure secrets management.
    """
    try:
        return st.secrets["api_keys"]["steam"]
    except KeyError:
        st.error("Steam API key not found in secrets. Please add your Steam API key to the secrets.toml file.")
        st.info("Create a .streamlit/secrets.toml file with:\n\n[api_keys]\nsteam = \"YOUR_STEAM_API_KEY\"")
        return None
    except Exception as e:
        st.error(f"Error accessing Steam API key: {e}")
        return None

def validate_api_key(api_key):
    """
    Basic validation for Steam API key format.
    Steam API keys are typically 32-character hexadecimal strings.
    """
    if not api_key:
        return False

    # Remove whitespace
    api_key = api_key.strip()

    # Check length (Steam API keys are usually 32 characters)
    if len(api_key) != 32:
        return False

    # Check if it's hexadecimal
    try:
        int(api_key, 16)
        return True
    except ValueError:
        return False

def format_playtime(minutes):
    """
    Convert playtime from minutes to a human-readable format.
    """
    if minutes == 0:
        return "Never played"

    hours = minutes / 60

    if hours < 1:
        return f"{minutes} minutes"
    elif hours < 24:
        return f"{hours:.1f} hours"
    else:
        days = hours / 24
        return f"{days:.1f} days ({hours:.0f} hours)"

def safe_get_game_name(game_data, default="Unknown Game"):
    """
    Safely extract game name from Steam API response.
    """
    if isinstance(game_data, dict):
        return game_data.get('name', default)
    return default

def clean_steam_url(url):
    """
    Clean and validate Steam profile URLs.
    """
    if not url:
        return None

    url = url.strip()

    # Add protocol if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    # Validate Steam domain
    if 'steamcommunity.com' not in url:
        return None

    return url
