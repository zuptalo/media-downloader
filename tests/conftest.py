import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add the app directory to the Python path
app_path = Path(__file__).parent.parent
sys.path.insert(0, str(app_path))

from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application"""
    return TestClient(app)


@pytest.fixture
def mock_video_info():
    """Fixture providing mock video information"""
    return {
        'title': 'Test Video',
        'formats': [
            {'height': 1080, 'filesize': 1024000, 'vcodec': 'avc1', 'acodec': 'mp4a'},
            {'height': 720, 'filesize': 512000, 'vcodec': 'avc1', 'acodec': 'mp4a'},
            {'vcodec': 'none', 'acodec': 'mp4a', 'filesize': 102400},
        ],
        'thumbnail': 'https://example.com/thumb.jpg',
        'duration': 300,
        'is_live': False
    }
