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
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
)
logger = logging.getLogger(__name__)

# Create a thread pool for handling downloads
download_executor = ThreadPoolExecutor(max_workers=3)

app = FastAPI(
    title="Media Download API",
    debug=os.getenv("DEBUG", "false").lower() == "true"
)


class DownloadRequest(BaseModel):
    urls: List[str]


class Format(BaseModel):
    format_id: str
    display_name: str


class MediaInfo(BaseModel):
    url: str
    title: str
    formats: List[Format]
    thumbnail: Optional[str]
    duration: Optional[int] = None
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


def format_size(size_in_bytes: Optional[int]) -> str:
    """Convert bytes to human readable string."""
    if size_in_bytes is None:
        return "Size unknown"

    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.1f} GB"


def estimate_format_size(formats: List[dict], target_height: Optional[int], audio_only: bool = False) -> Optional[str]:
    """Estimate size for a given quality target."""
    if audio_only:
        # Find best audio format
        audio_formats = [f for f in formats if f.get('vcodec') == 'none' and f.get('acodec') != 'none']
        if audio_formats:
            best_audio = max(audio_formats, key=lambda x: x.get('filesize', 0) or 0)
            return format_size(best_audio.get('filesize'))
        return None

    if target_height:
        # Find video format closest to target height, safely handling None heights
        video_formats = [
            f for f in formats
            if f.get('height') is not None
               and f.get('height') <= target_height
               and f.get('vcodec') != 'none'
        ]

        if video_formats:
            # Get the best quality video within height limit
            best_video = max(video_formats, key=lambda x: (x.get('height', 0), x.get('filesize', 0) or 0))
            video_size = best_video.get('filesize', 0) or 0

            # Add audio size
            audio_formats = [f for f in formats if f.get('vcodec') == 'none' and f.get('acodec') != 'none']
            if audio_formats:
                best_audio = max(audio_formats, key=lambda x: x.get('filesize', 0) or 0)
                audio_size = best_audio.get('filesize', 0) or 0
                total_size = video_size + audio_size
                return format_size(total_size)

            return format_size(video_size)
    return None


async def get_video_info(url: str) -> MediaInfo:
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            is_live = info.get('is_live', False)
            available_formats = info.get('formats', [])

            if is_live:
                formats = [
                    Format(
                        format_id="best",
                        display_name="Best Quality (Live Stream)"
                    ),
                    Format(
                        format_id="worst",
                        display_name="Low Quality (Live Stream)"
                    )
                ]
            else:
                formats = []
                quality_options = [
                    (2160, "4K Quality (2160p)"),
                    (1440, "2K Quality (1440p)"),
                    (1080, "Full HD (1080p)"),
                    (720, "HD (720p)"),
                    (480, "SD (480p)")
                ]

                # Add video quality options that have available formats
                for height, name in quality_options:
                    size = estimate_format_size(available_formats, height)
                    if size and size != "Size unknown":
                        formats.append(Format(
                            format_id=f"bv*[height<={height}]+ba/b[height<={height}]",
                            display_name=f"{name} - {size}"
                        ))

                # Always add audio option
                audio_size = estimate_format_size(available_formats, None, audio_only=True)
                if audio_size:
                    formats.append(Format(
                        format_id="ba/b",
                        display_name=f"Audio Only - {audio_size}"
                    ))

            # Sort formats by quality (highest first)
            formats = sorted(formats,
                             key=lambda x: (0 if "Audio" in x.display_name else
                                            int(x.format_id.split("<=")[1].split("]")[0])
                                            if "<=" in x.format_id else 999999),
                             reverse=True)

            return MediaInfo(
                url=url,
                title=info.get('title', ''),
                formats=formats,
                thumbnail=info.get('thumbnail'),
                duration=info.get('duration'),
                is_live=is_live
            )

    except Exception as e:
        logger.error(f"Error extracting info: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error extracting info: {str(e)}")


@app.post("/analyze", response_model=List[MediaInfo])
async def analyze_urls(request: DownloadRequest):
    tasks = [get_video_info(url) for url in request.urls]
    return await asyncio.gather(*tasks)


def download_file(url: str, format_id: str, embed_thumbnail: bool, duration: Optional[int], temp_dir: str) -> tuple[
    str, str]:
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

    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    env = os.getenv("ENVIRONMENT", "development")

    reload_enabled = env == "development"

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload_enabled,
        reload_dirs=["app"] if reload_enabled else None,
        log_level="debug" if debug else "info"
    )
