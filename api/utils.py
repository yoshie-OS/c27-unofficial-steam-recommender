# from cryptography.fernet import Fernet
import streamlit as st

def decrypt_api_key():
    """Decrypt the API key using the encryption key"""
    # Initially, this function was used to decrypt the API code
    # which was stored in a .bin for the encrypted key and .key file for the encryption key.
    # I kept the name 'decrypt_api_key' in order to avoid having to change the name of the function in each file
    # that has one or more variables in which this function is assigned to.

    try:
        return st.secrets["api_keys"]["steam"]
    except Exception as e:
        st.error(f"Error accessing Steam API key: {e}")
        return None
