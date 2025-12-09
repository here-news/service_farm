"""
Google OAuth configuration
Copied from webapp/app/auth/google_oauth.py
"""

import os
from authlib.integrations.starlette_client import OAuth

# Load settings from environment
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', '')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET', '')

# Initialize OAuth
oauth = OAuth()

oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


def get_google_oauth():
    """Get configured Google OAuth client"""
    return oauth.google
