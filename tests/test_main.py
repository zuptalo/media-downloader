from unittest.mock import patch, MagicMock, call

import pytest
from yt_dlp.utils import DownloadError

from app.main import sanitize_filename, estimate_format_size


# Unit tests for utility functions
def test_sanitize_filename():
    """Test filename sanitization function"""
    # Test basic sanitization
    assert sanitize_filename('test.mp4') == 'test.mp4'

    # Test invalid characters
    assert sanitize_filename('test/file:with*invalid<chars>.mp4') == 'testfilewithinvalidchars.mp4'

    # Test spaces
    assert sanitize_filename('  test  file  ') == 'test file'

    # Test unicode characters
    assert sanitize_filename('tést vídéo.mp4').replace(' ', '') == 'testvideo.mp4'.replace(' ', '')


def test_estimate_format_size():
    """Test format size estimation function"""
    test_formats = [
        {'height': 1080, 'filesize': 1024000, 'vcodec': 'avc1', 'acodec': 'mp4a'},
        {'height': 720, 'filesize': 512000, 'vcodec': 'avc1', 'acodec': 'mp4a'},
        {'vcodec': 'none', 'acodec': 'mp4a', 'filesize': 102400},
    ]

    # Test video format estimation
    size = estimate_format_size(test_formats, 1080)
    assert size is not None
    assert "MB" in size or "KB" in size

    # Test audio-only estimation
    audio_size = estimate_format_size(test_formats, None, audio_only=True)
    assert audio_size is not None
    assert "KB" in audio_size

    # Test with empty formats list
    no_size = estimate_format_size([], 1080)
    assert no_size is None

    # Test with formats but no filesize
    formats_no_size = [
        {'height': 1080, 'vcodec': 'avc1', 'acodec': 'mp4a'},  # No filesize
        {'vcodec': 'none', 'acodec': 'mp4a'},  # No filesize
    ]
    size_unknown = estimate_format_size(formats_no_size, 1080)
    assert size_unknown == "Size unknown"


@pytest.mark.asyncio
async def test_analyze_endpoint(client, mock_video_info):
    """Test the /analyze endpoint"""
    with patch('yt_dlp.YoutubeDL') as mock_ydl:
        # Configure mock
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = mock_video_info
        mock_ydl.return_value.__enter__.return_value = mock_instance

        # Test endpoint
        response = client.post(
            "/analyze",
            json={"urls": ["https://www.youtube.com/watch?v=test"]}
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["title"] == "Test Video"
        assert len(data[0]["formats"]) > 0


@pytest.mark.asyncio
async def test_analyze_endpoint_error(client):
    """Test error handling in /analyze endpoint"""
    with patch('yt_dlp.YoutubeDL') as mock_ydl:
        # Configure mock to raise an exception
        mock_instance = MagicMock()
        mock_instance.extract_info.side_effect = Exception("Download error")
        mock_ydl.return_value.__enter__.return_value = mock_instance

        # Test endpoint
        response = client.post(
            "/analyze",
            json={"urls": ["https://www.youtube.com/watch?v=invalid"]}
        )

        assert response.status_code == 400
        assert "error" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_analyze_non_url_download_error(client):
    with patch('yt_dlp.YoutubeDL') as mock_ydl:
        mock_instance = MagicMock()
        mock_instance.extract_info.side_effect = DownloadError("Some other extraction error")
        mock_ydl.return_value.__enter__.return_value = mock_instance

        response = client.post("/analyze", json={"urls": ["https://example.com/test"]})

        # Now we expect 200 with downloadable=False since we're no longer raising HTTPException
        assert response.status_code == 200
        data = response.json()
        assert data[0]["downloadable"] is False
        assert data[0]["title"] == ""
        assert data[0]["formats"] == []


@pytest.mark.asyncio
async def test_analyze_generic_exception(client):
    """Test that a generic exception triggers the generic except block."""
    with patch('yt_dlp.YoutubeDL') as mock_ydl:
        mock_instance = MagicMock()

        # Trigger a generic Exception
        mock_instance.extract_info.side_effect = Exception("Unexpected error")
        mock_ydl.return_value.__enter__.return_value = mock_instance

        response = client.post("/analyze", json={"urls": ["https://example.com/test"]})

        # Now we expect 200 with downloadable=False since we're no longer raising HTTPException
        assert response.status_code == 200
        data = response.json()
        assert data[0]["downloadable"] is False
        assert data[0]["title"] == ""
        assert data[0]["formats"] == []


@pytest.mark.asyncio
async def test_download_endpoint(client):
    """Test the /download endpoint"""
    mock_content = b"test video content"
    test_filename = "test_video.mp4"

    with patch('yt_dlp.YoutubeDL') as mock_ydl, \
            patch('builtins.open', create=True) as mock_open, \
            patch('shutil.rmtree') as mock_rmtree, \
            patch('tempfile.mkdtemp', return_value='/tmp/test'), \
            patch('os.path.exists') as mock_exists:
        # Configure mocks
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = {
            'title': 'Test Video',
            'is_live': False
        }
        mock_instance.prepare_filename.return_value = test_filename
        mock_ydl.return_value.__enter__.return_value = mock_instance

        # Mock file existence check
        mock_exists.return_value = True

        # Mock file operations
        mock_file = MagicMock()
        mock_file.read.return_value = mock_content
        mock_open.return_value.__enter__.return_value = mock_file

        # Test endpoint
        response = client.post(
            "/download",
            json={
                "url": "https://www.youtube.com/watch?v=test",
                "format_id": "bv*[height<=1080]+ba/b[height<=1080]",
                "embed_thumbnail": True
            }
        )

        assert response.status_code == 200
        assert response.content == mock_content
        assert response.headers['content-type'] == 'video/mp4'
        assert 'content-disposition' in response.headers
        assert mock_rmtree.called  # Verify cleanup was attempted


@pytest.mark.asyncio
async def test_download_endpoint_live_stream(client):
    """Test downloading a live stream"""
    mock_content = b"test live stream content"
    test_filename = "test_live.mp4"

    with patch('yt_dlp.YoutubeDL') as mock_ydl, \
            patch('builtins.open', create=True) as mock_open, \
            patch('shutil.rmtree') as mock_rmtree, \
            patch('tempfile.mkdtemp', return_value='/tmp/test'), \
            patch('os.path.exists') as mock_exists:
        # Configure mocks
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = {
            'title': 'Test Live Stream',
            'is_live': True
        }
        mock_instance.prepare_filename.return_value = test_filename
        mock_ydl.return_value.__enter__.return_value = mock_instance

        mock_exists.return_value = True
        mock_file = MagicMock()
        mock_file.read.return_value = mock_content
        mock_open.return_value.__enter__.return_value = mock_file

        response = client.post(
            "/download",
            json={
                "url": "https://www.youtube.com/watch?v=live",
                "format_id": "best",
                "embed_thumbnail": False,
                "duration": 30
            }
        )

        assert response.status_code == 200
        assert response.content == mock_content


@pytest.mark.asyncio
async def test_live_stream_handling(client, mock_video_info):
    """Test handling of live stream content"""
    with patch('yt_dlp.YoutubeDL') as mock_ydl:
        # Configure mock for live stream
        live_info = {**mock_video_info, 'is_live': True}
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = live_info
        mock_ydl.return_value.__enter__.return_value = mock_instance

        response = client.post(
            "/analyze",
            json={"urls": ["https://www.youtube.com/watch?v=live"]}
        )

        assert response.status_code == 200
        data = response.json()
        assert data[0]["is_live"] == True
        formats = [f["format_id"] for f in data[0]["formats"]]
        assert "best" in formats
        assert "worst" in formats


@pytest.mark.asyncio
async def test_download_endpoint_file_not_found(client):
    """Test download endpoint when file is not found after download"""
    with patch('yt_dlp.YoutubeDL') as mock_ydl, \
            patch('tempfile.mkdtemp', return_value='/tmp/test'), \
            patch('os.path.exists') as mock_exists:
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = {
            'title': 'Test Video',
            'is_live': False
        }
        mock_instance.prepare_filename.return_value = "nonexistent.mp4"
        mock_ydl.return_value.__enter__.return_value = mock_instance

        # Mock file not existing
        mock_exists.return_value = False

        response = client.post(
            "/download",
            json={
                "url": "https://www.youtube.com/watch?v=test",
                "format_id": "best",
                "embed_thumbnail": True
            }
        )

        assert response.status_code == 400
        assert "file not found" in response.json()["detail"].lower()


def test_format_size():
    """Test format size utility function"""
    from app.main import format_size

    # Test various size ranges
    assert format_size(500) == "500.0 B"
    assert format_size(1500) == "1.5 KB"
    assert format_size(1500000) == "1.4 MB"
    assert format_size(1500000000) == "1.4 GB"
    assert format_size(None) == "Size unknown"


@pytest.mark.asyncio
async def test_download_endpoint_live_with_duration(client):
    """Test downloading a live stream with duration limit"""
    mock_content = b"test live content"
    test_filename = "test_live.mp4"

    with patch('yt_dlp.YoutubeDL') as mock_ydl, \
            patch('builtins.open', create=True) as mock_open, \
            patch('shutil.rmtree') as mock_rmtree, \
            patch('tempfile.mkdtemp', return_value='/tmp/test'), \
            patch('os.path.exists') as mock_exists:
        # Configure mocks
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = {
            'title': 'Test Live Stream',
            'is_live': True
        }
        mock_instance.prepare_filename.return_value = test_filename
        mock_ydl.return_value.__enter__.return_value = mock_instance

        mock_exists.return_value = True
        mock_file = MagicMock()
        mock_file.read.return_value = mock_content
        mock_open.return_value.__enter__.return_value = mock_file

        # Test live stream download with duration
        response = client.post(
            "/download",
            json={
                "url": "https://www.youtube.com/watch?v=live",
                "format_id": "best",
                "duration": 30
            }
        )

        assert response.status_code == 200
        assert response.content == mock_content

        # Verify that extract_info was called twice (once for info, once for download)
        assert mock_instance.extract_info.call_count == 2

        # Check the calls were made correctly
        mock_instance.extract_info.assert_has_calls([
            call("https://www.youtube.com/watch?v=live", download=False),
            call("https://www.youtube.com/watch?v=live", download=True)
        ])


@pytest.mark.asyncio
async def test_download_endpoint_with_exception(client):
    """Test download endpoint when an exception occurs during download"""
    with patch('yt_dlp.YoutubeDL') as mock_ydl, \
            patch('tempfile.mkdtemp', return_value='/tmp/test'):
        mock_instance = MagicMock()
        mock_instance.extract_info.side_effect = Exception("Download failed")
        mock_ydl.return_value.__enter__.return_value = mock_instance

        response = client.post(
            "/download",
            json={
                "url": "https://www.youtube.com/watch?v=test",
                "format_id": "best",
                "embed_thumbnail": True
            }
        )

        assert response.status_code == 400
        assert "download failed" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_url_downloadable_check(client):
    """Test the URL downloadability check"""
    with patch('yt_dlp.YoutubeDL') as mock_ydl, \
            patch('yt_dlp.extractor.gen_extractors') as mock_gen_extractors:
        # Mock extractor that considers YouTube URLs suitable
        mock_extractor = MagicMock()
        mock_extractor.suitable.side_effect = lambda url: 'youtube.com' in url
        mock_extractor.IE_NAME = 'youtube'

        # Mock extractors list
        mock_gen_extractors.return_value = [mock_extractor]

        # Configure YoutubeDL mock for successful case
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = {
            'title': 'Test Video',
            'formats': [],
            'is_live': False
        }
        mock_ydl.return_value.__enter__.return_value = mock_instance

        # Test downloadable URL (YouTube)
        response = client.post(
            "/analyze",
            json={"urls": ["https://www.youtube.com/watch?v=test"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert data[0]["downloadable"] == True

        # Test non-downloadable URL
        response = client.post(
            "/analyze",
            json={"urls": ["https://example.com/not-a-video"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert data[0]["downloadable"] == False


@pytest.mark.asyncio
async def test_downloadable_errors(client):
    """Test error handling in downloadability check"""
    with patch('yt_dlp.extractor.gen_extractors') as mock_gen_extractors:
        # Mock extractor that raises an exception
        mock_extractor = MagicMock()
        mock_extractor.suitable.side_effect = Exception("Extractor error")

        # Mock extractors list
        mock_gen_extractors.return_value = [mock_extractor]

        response = client.post("/analyze", json={"urls": ["https://example.com/test"]})

        # Now that we no longer raise a 400 error, we expect a 200 with downloadable=False
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]['downloadable'] is False
        assert data[0]['title'] == ""
        assert data[0]['formats'] == []


@pytest.mark.asyncio
async def test_analyze_invalid_url(client):
    """Test the /analyze endpoint with a string that's not a valid URL"""
    # No special mocking required here if you rely on actual yt_dlp behavior.
    # If you prefer to mock yt_dlp responses, you could mock them similarly
    # to other tests. For simplicity, this test will rely on the real behavior.

    response = client.post("/analyze", json={"urls": ["Clipboard 8 Dec 2024 at 06.44"]})
    assert response.status_code == 200

    data = response.json()
    assert len(data) == 1
    assert data[0]["downloadable"] is False
    assert data[0]["title"] == ""
    assert data[0]["formats"] == []
    # Optional: Check for other fields if needed.
