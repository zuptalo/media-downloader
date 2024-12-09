import os
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from yt_dlp.utils import DownloadError

from app.main import app, DeviceType, sanitize_filename, create_quality_label, detect_device_type


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
            {
                'format_id': 'high',
                'height': 1080,
                'width': 1920,
                'filesize': 1024000,
                'vcodec': 'avc1',
                'acodec': 'mp4a',
                'fps': 30,
                'tbr': 2500,
                'ext': 'mp4'
            },
            {
                'format_id': 'low',
                'height': 720,
                'width': 1280,
                'filesize': 512000,
                'vcodec': 'avc1',
                'acodec': 'mp4a',
                'fps': 30,
                'tbr': 1500,
                'ext': 'mp4'
            },
            {
                'format_id': 'audio',
                'vcodec': 'none',
                'acodec': 'mp4a',
                'filesize': 102400,
                'ext': 'm4a'
            }
        ],
        'thumbnail': 'https://example.com/thumb.jpg',
        'duration': 300,
        'is_live': False
    }


def test_sanitize_filename():
    """Test filename sanitization for various cases"""
    # Test basic sanitization
    assert sanitize_filename('test.mp4') == 'test.mp4'

    # Test invalid characters
    assert sanitize_filename('test/file:with*invalid<chars>.mp4') == 'testfilewithinvalidchars.mp4'

    # Test spaces
    assert sanitize_filename('  test  file  ') == 'test file'

    # Test unicode characters
    assert sanitize_filename('tést vídéo.mp4').replace(' ', '') == 'testvideo.mp4'.replace(' ', '')

    # Test long filenames (should truncate to 200 chars)
    long_name = 'a' * 250 + '.mp4'
    assert len(sanitize_filename(long_name)) == 200


def test_create_quality_label():
    """Test quality label creation for different formats"""
    # Test full HD format
    hd_format = {
        'height': 1080,
        'width': 1920,
        'fps': 30,
        'vcodec': 'avc1.123456',
        'acodec': 'mp4a.40.2',
        'filesize': 1024 * 1024 * 10,  # 10MB
        'tbr': 2500
    }
    label = create_quality_label(hd_format)
    assert '1080p' in label
    assert 'Full HD' in label
    assert '[h264/aac]' in label  # Updated assertion

    # Test 4K HDR format
    hdr_format = {
        'height': 2160,
        'width': 3840,
        'fps': 60,
        'vcodec': 'vp09.02.51.12.01.09.16.09.01',
        'acodec': 'opus',
        'dynamic_range': 'HDR',
        'format_note': 'HDR'
    }
    label = create_quality_label(hdr_format)
    assert '2160p' in label
    assert '4K' in label
    assert 'HDR' in label
    assert '60fps' in label

    # Test audio-only format
    audio_format = {
        'vcodec': 'none',
        'acodec': 'mp4a.40.2',
        'filesize': 1024 * 1024
    }
    label = create_quality_label(audio_format)
    assert '[aac]' in label  # Updated assertion


@pytest.mark.parametrize(
    "user_agent,expected_type",
    [
        ("Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)", DeviceType.MOBILE),
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64)", DeviceType.DESKTOP),
        (None, DeviceType.DESKTOP),  # Default case
    ]
)
def test_detect_device_type(user_agent, expected_type):
    """Test device type detection from user agent strings"""
    assert detect_device_type(user_agent) == expected_type


@pytest.mark.asyncio
async def test_analyze_endpoint(client, mock_video_info):
    """Test the /analyze endpoint for various scenarios"""
    with patch('yt_dlp.YoutubeDL') as mock_ydl:
        # Configure mock
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = mock_video_info
        mock_ydl.return_value.__enter__.return_value = mock_instance

        # Test with single URL
        response = client.post(
            "/analyze",
            json={"urls": ["https://www.youtube.com/watch?v=test"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["title"] == "Test Video"
        assert len(data[0]["formats"]) > 0

        # Test with multiple URLs
        response = client.post(
            "/analyze",
            json={"urls": [
                "https://www.youtube.com/watch?v=test1",
                "https://www.youtube.com/watch?v=test2"
            ]}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2


@pytest.mark.asyncio
async def test_analyze_live_content(client):
    """Test analyzing live stream content"""
    live_info = {
        'title': 'Test Live Stream',
        'formats': [
            {
                'format_id': 'live_high',
                'height': 1080,
                'width': 1920,
                'vcodec': 'avc1',
                'acodec': 'mp4a',
                'tbr': 5000,
                'is_live': True
            }
        ],
        'is_live': True
    }

    with patch('yt_dlp.YoutubeDL') as mock_ydl:
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = live_info
        mock_ydl.return_value.__enter__.return_value = mock_instance

        response = client.post(
            "/analyze",
            json={"urls": ["https://www.youtube.com/watch?v=live"]}
        )

        assert response.status_code == 200
        data = response.json()
        assert data[0]["is_live"] is True
        assert any('Live Stream' in f['display_name'] for f in data[0]['formats'])


@pytest.mark.asyncio
async def test_download_endpoint(client):
    """Test the /download endpoint"""
    mock_content = b"test video content"
    test_filename = "test_video.mp4"
    temp_dir = "/tmp/test"

    # Create a single format that will work for both analysis and download
    test_format = {
        'format_id': 'test',
        'ext': 'mp4',
        'vcodec': 'avc1',
        'acodec': 'mp4a',
        'height': 1080,
        'width': 1920,
        'tbr': 2500,
        'filesize': 1024000,
        'format': 'test - 1080p'
    }

    analysis_info = {
        'title': 'Test Video',
        'formats': [test_format],
        'is_live': False,
        'duration': 300,
        'thumbnail': 'https://example.com/thumb.jpg',
        'webpage_url': 'https://www.youtube.com/watch?v=test'
    }

    download_info = {
        **analysis_info,
        'requested_downloads': [{
            **test_format,
            '_filename': os.path.join(temp_dir, test_filename)
        }]
    }

    with patch('yt_dlp.YoutubeDL') as mock_ydl, \
            patch('builtins.open', create=True) as mock_open, \
            patch('shutil.rmtree') as mock_rmtree, \
            patch('tempfile.mkdtemp', return_value=temp_dir), \
            patch('os.path.exists', return_value=True):
        # Configure mock for both analyze and download calls
        mock_instance = MagicMock()
        mock_instance.extract_info.side_effect = [analysis_info, analysis_info,
                                                  download_info]  # Allow for two analysis calls
        mock_instance.prepare_filename.return_value = os.path.join(temp_dir, test_filename)
        mock_ydl.return_value.__enter__.return_value = mock_instance

        # Mock file operations
        mock_file = MagicMock()
        mock_file.read.return_value = mock_content
        mock_open.return_value.__enter__.return_value = mock_file

        # Test standard download
        response = client.post(
            "/download",
            json={
                "url": "https://www.youtube.com/watch?v=test",
                "format_id": "test",
                "embed_thumbnail": True
            }
        )

        # Verify HTTP response
        assert response.status_code == 200
        assert response.content == mock_content
        assert response.headers['content-type'] == 'video/mp4'
        assert 'content-disposition' in response.headers

        # Verify yt-dlp interactions
        extract_info_calls = mock_instance.extract_info.call_args_list

        # Verify we have the correct sequence of calls
        assert len(extract_info_calls) == 3, "Should have three extract_info calls in total"

        # Verify the analysis calls (first two should be download=False)
        assert extract_info_calls[0][1]['download'] is False
        assert extract_info_calls[1][1]['download'] is False

        # Verify the download call (last one should be download=True)
        assert extract_info_calls[2][1]['download'] is True

        # Verify file cleanup
        assert mock_rmtree.called


@pytest.mark.asyncio
async def test_download_errors(client):
    """Test error handling in download endpoint"""
    with patch('yt_dlp.YoutubeDL') as mock_ydl:
        # Test unsupported URL
        mock_instance = MagicMock()
        mock_instance.extract_info.side_effect = DownloadError("Unsupported URL")
        mock_ydl.return_value.__enter__.return_value = mock_instance

        response = client.post(
            "/download",
            json={
                "url": "https://example.com/unsupported",
                "format_id": "test"
            }
        )

        assert response.status_code == 400
        assert "url is not supported for download" in response.json()["detail"].lower()  # Updated assertion


@pytest.mark.asyncio
async def test_device_specific_formats(client, mock_video_info):
    """Test format selection based on device type"""
    with patch('yt_dlp.YoutubeDL') as mock_ydl:
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = mock_video_info
        mock_ydl.return_value.__enter__.return_value = mock_instance

        # Test mobile device
        response = client.post(
            "/analyze",
            json={
                "urls": ["https://www.youtube.com/watch?v=test"],
                "device_type": "mobile"
            }
        )

        assert response.status_code == 200
        data = response.json()
        formats = data[0]["formats"]

        # Verify format constraints for mobile
        for fmt in formats:
            if 'height' in fmt and fmt['height'] is not None:  # Check for None
                assert fmt['height'] <= 1080  # Mobile max resolution
            if 'fps' in fmt and fmt['fps'] is not None:  # Check for None
                assert fmt['fps'] <= 30  # Mobile max fps

        # Test desktop device
        response = client.post(
            "/analyze",
            json={
                "urls": ["https://www.youtube.com/watch?v=test"],
                "device_type": "desktop"
            }
        )

        assert response.status_code == 200
        data = response.json()
        formats = data[0]["formats"]

        # Verify higher quality formats are available for desktop
        heights = [f['height'] for f in formats if f.get('height') is not None]
        if heights:  # Only assert if we have valid heights
            max_height = max(heights)
            assert max_height >= 1080  # Should allow higher resolutions
