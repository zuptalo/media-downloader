import asyncio
import logging
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import yt_dlp
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import Response
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a thread pool for handling downloads
download_executor = ThreadPoolExecutor(max_workers=3)  # Adjust number based on server capacity

app = FastAPI(title="Media Download API")


class DownloadRequest(BaseModel):
    urls: List[str]


class Format(BaseModel):
    format_id: str
    ext: str
    quality: Optional[str | int | float] = None
    filesize: Optional[int] = None
    format_note: Optional[str] = None
    acodec: Optional[str] = None
    vcodec: Optional[str] = None


class MediaInfo(BaseModel):
    url: str
    title: str
    formats: List[Format]
    thumbnail: Optional[str]
    channel: Optional[str] = None
    upload_date: Optional[str] = None
    description: Optional[str] = None
    is_live: Optional[bool] = False


class DownloadFormat(BaseModel):
    url: str
    format_id: str
    embed_thumbnail: bool = True
    duration: Optional[int] = None  # Duration in seconds for live streams


def sanitize_filename(title: str) -> str:
    """Sanitize the filename to remove invalid characters."""
    title = re.sub(r'[<>:"/\\|?*]', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title[:200]


async def get_video_info(url: str) -> MediaInfo:
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            formats = [
                Format(
                    format_id=str(f['format_id']),
                    ext=str(f.get('ext', '')),
                    quality=f.get('quality'),
                    filesize=f.get('filesize'),
                    format_note=str(f.get('format_note', '')),
                    acodec=str(f.get('acodec', '')),
                    vcodec=str(f.get('vcodec', ''))
                )
                for f in info.get('formats', [])
            ]

            return MediaInfo(
                url=url,
                title=info.get('title', ''),
                formats=formats,
                thumbnail=info.get('thumbnail'),
                channel=info.get('channel', ''),
                upload_date=info.get('upload_date', ''),
                description=info.get('description', ''),
                is_live=info.get('is_live', False)
            )
    except Exception as e:
        logger.error(f"Error extracting info: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error extracting info: {str(e)}")


@app.post("/analyze", response_model=List[MediaInfo])
async def analyze_urls(request: DownloadRequest):
    tasks = [get_video_info(url) for url in request.urls]
    return await asyncio.gather(*tasks)


def download_file(url: str, format_id: str, embed_thumbnail: bool, duration: Optional[int], temp_dir: str) -> tuple[str, str]:
    """Download file in a separate thread."""
    try:
        # Get video info first to get the title
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            safe_title = sanitize_filename(info.get('title', ''))
            is_live = info.get('is_live', False)

        # Configure postprocessors
        postprocessors = [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }]

        if embed_thumbnail and info.get('thumbnail'):
            postprocessors.append({
                'key': 'EmbedThumbnail',
                'already_have_thumbnail': False,
            })

        # For live streams with duration, we'll use ffmpeg directly via yt-dlp
        if is_live and duration:
            ydl_opts = {
                'format': 'best',  # Use best single format for live streams
                'outtmpl': str(Path(temp_dir) / f'{safe_title}.%(ext)s'),
                'external_downloader': 'ffmpeg',
                'external_downloader_args': {
                    'ffmpeg_i': ['-t', str(duration)]  # Direct duration limit for ffmpeg
                },
                'postprocessors': postprocessors,
                'writethumbnail': embed_thumbnail,
                'verbose': True,
            }
        else:
            ydl_opts = {
                'format': f'{format_id}+bestaudio/best',
                'outtmpl': str(Path(temp_dir) / f'{safe_title}.%(ext)s'),
                'merge_output_format': 'mp4',
                'postprocessors': postprocessors,
                'writethumbnail': embed_thumbnail,
                'verbose': True,
            }

        # Download the file
        start_time = time.time()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Starting download for format {format_id}")
            download_info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(download_info)

        # Handle potential filename extension change after postprocessing
        if not os.path.exists(filename):
            mp4_filename = str(Path(filename).with_suffix('.mp4'))
            if os.path.exists(mp4_filename):
                filename = mp4_filename
            else:
                raise FileNotFoundError("File not found after download")

        output_filename = f"{safe_title}.mp4"
        return filename, output_filename

    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise


@app.post("/download")
async def download_media(download_request: DownloadFormat, background_tasks: BackgroundTasks):
    # Create a temporary directory that will persist until we're done
    temp_dir = tempfile.mkdtemp()

    try:
        # Execute download in thread pool
        filename, output_filename = await asyncio.get_event_loop().run_in_executor(
            download_executor,
            download_file,
            download_request.url,
            download_request.format_id,
            download_request.embed_thumbnail,
            download_request.duration,
            temp_dir
        )

        # Read the file content
        with open(filename, 'rb') as f:
            file_content = f.read()

        async def cleanup_temp_dir():
            try:
                await asyncio.sleep(1)  # Small delay to ensure file is served
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory: {str(e)}")

        background_tasks.add_task(cleanup_temp_dir)

        return Response(
            content=file_content,
            media_type='video/mp4',
            headers={
                'Content-Disposition': f'attachment; filename="{output_filename}"',
                'Content-Length': str(len(file_content))
            }
        )

    except Exception as e:
        # Clean up temp directory in case of any errors
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["app"],
        log_level="debug"
    )
