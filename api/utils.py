# from cryptography.fernet import Fernet
import streamlit as st

def decrypt_api_key():
    """Decrypt the API key using the encryption key"""
    # # Load the encryption key
    # with open('./.streamlit/encryption_key.key', 'rb') as key_file:
    #     encryption_key = key_file.read()

    # # Load the encrypted API key
    # with open('./.streamlit/encrypted_api_key.bin', 'rb') as file:
    #     encrypted_api_key = file.read()

    # # Decrypt the API key
    # fernet = Fernet(encryption_key)
    # api_key = fernet.decrypt(encrypted_api_key).decode()

    try:
        return st.secrets["api_keys"]["steam"]
    except Exception as e:
        st.error(f"Error accessing Steam API key: {e}")
        return None
