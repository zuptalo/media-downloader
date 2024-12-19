import asyncio
import logging
import os
import re
import shutil
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import yt_dlp
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from fastapi.responses import Response
from pydantic import BaseModel
from user_agents import parse

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG", "true").lower() == "true" else logging.INFO
)
logger = logging.getLogger(__name__)

# Create a thread pool for handling downloads
download_executor = ThreadPoolExecutor(max_workers=3)

app = FastAPI(
    title="Media Download API",
    description="API for analyzing and downloading media from various platforms",
    version="1.0.0",
    debug=os.getenv("DEBUG", "false").lower() == "true"
)


class DeviceType(str, Enum):
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TV = "tv"


class DownloadRequest(BaseModel):
    urls: List[str]
    device_type: Optional[DeviceType] = None
    force_quality: Optional[str] = None


class FormatInfo(BaseModel):
    format_id: str
    ext: str
    quality: str
    filesize: Optional[int]
    vcodec: Optional[str]
    acodec: Optional[str]
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    tbr: Optional[float]
    compatibility_score: int = 0
    is_recommended: bool = False
    display_name: str
    error: Optional[str] = None


class MediaAnalysis(BaseModel):
    url: str
    title: str
    downloadable: bool
    formats: List[FormatInfo]
    is_live: bool
    duration: Optional[float] = None
    thumbnail: Optional[str]


class DownloadFormat(BaseModel):
    url: str
    format_id: str
    device_type: Optional[DeviceType] = None
    embed_thumbnail: bool = True
    duration: Optional[int] = None


DEVICE_CONSTRAINTS = {
    DeviceType.MOBILE: {
        "max_height": 1080,
        "max_fps": 30,
        "preferred_codecs": ["h264", "avc1"],
        "max_size_mb": 500,
        "max_bitrate": 5000  # kbps
    },
    DeviceType.DESKTOP: {
        "max_height": 2160,
        "max_fps": 60,
        "preferred_codecs": ["h264", "avc1", "vp9", "av1"],
        "max_size_mb": None,
        "max_bitrate": None
    },
    DeviceType.TV: {
        "max_height": 2160,
        "max_fps": 60,
        "preferred_codecs": ["h264", "avc1", "hevc"],
        "max_size_mb": None,
        "max_bitrate": 25000  # kbps
    }
}


def sort_format_key(fmt: Dict) -> Tuple[int, bool, int, float]:
    """Enhanced key function for sorting formats by quality."""
    if not fmt:
        return (0, False, 0, 0.0)

    # For format info objects
    if isinstance(fmt, FormatInfo):
        height = fmt.height if fmt.height is not None else 0
        has_hdr = bool(fmt.quality and 'HDR' in fmt.quality)
        fps = fmt.fps if fmt.fps is not None else 0
        size = float('inf') if fmt.filesize is None else fmt.filesize
    else:
        height = fmt.get('height', 0) if fmt.get('height') is not None else 0
        has_hdr = bool('HDR' in str(fmt.get('dynamic_range', '')) or
                       'HDR' in str(fmt.get('quality', '')))
        fps = fmt.get('fps', 0) if fmt.get('fps') is not None else 0
        size = float('inf') if fmt.get('filesize') is None else fmt.get('filesize', 0)

    # Modified codec scoring - make H.264 more competitive
    codec_score = 0
    vcodec = str(fmt.get('vcodec', '')).lower()

    # Score based on codec quality but keep H.264 competitive
    if 'av01' in vcodec:
        codec_score = 1.2  # Just slightly better than others
    elif 'vp9' in vcodec or 'vp09' in vcodec:
        codec_score = 1.1
    elif 'avc1' in vcodec or 'h264' in vcodec:
        codec_score = 1.0  # Base score - still very competitive

    # Bonus for having both video and audio
    has_both = bool(fmt.get('vcodec') != 'none' and fmt.get('acodec') != 'none')
    if has_both:
        codec_score += 0.5

    # DRC adjustment - penalize DRC versions slightly
    is_drc = '-drc' in str(fmt.get('format_id', ''))
    drc_penalty = 0.1 if is_drc else 0

    return (-height, not has_hdr, -fps, float(-codec_score + drc_penalty))


def create_quality_label(fmt: Dict | FormatInfo) -> str:
    """Create a unified quality label for both Dict and FormatInfo formats."""
    try:
        # Handle both Dict and FormatInfo inputs
        if isinstance(fmt, FormatInfo):
            height = fmt.height or 0
            width = fmt.width or 0
            fps = fmt.fps or 0
            vcodec = fmt.vcodec
            acodec = fmt.acodec
            filesize = fmt.filesize
            is_portrait = getattr(fmt, 'is_portrait', False)
        else:
            height = fmt.get('height', 0) or 0
            width = fmt.get('width', 0) or 0
            fps = fmt.get('fps', 0) or 0
            vcodec = fmt.get('vcodec')
            acodec = fmt.get('acodec')
            filesize = fmt.get('filesize')
            is_portrait = bool(width < height) if width and height else False

        components = []

        # Resolution and quality label
        if height >= 1080:
            components.append(f"{height}p (Full HD)")
        elif height >= 720:
            components.append(f"{height}p (HD)")
        else:
            components.append(f"{height}p")

        # Portrait indicator
        if is_portrait:
            components.append("Portrait")

        # Frame rate (if higher than 30)
        if fps > 30:
            components.append(f"{int(fps)}fps")

        # File size
        if filesize:
            size = filesize / (1024 * 1024)  # Convert to MB
            if size >= 1:
                components.append(f"({size:.1f} MB)")
            else:
                size_kb = filesize / 1024
                components.append(f"({size_kb:.1f} KB)")

        # Format type indicators
        has_video = bool(vcodec and vcodec != 'none')
        has_audio = bool(acodec and acodec != 'none')

        if has_video and not has_audio:
            components.append("- Video Only")

        if has_video:
            if 'avc1' in str(vcodec).lower() or 'h264' in str(vcodec).lower():
                components.append("[h264]")
            elif 'av01' in str(vcodec).lower():
                components.append("[av1]")

        if has_audio and not has_video:
            components.append("- Audio Only")
            if 'opus' in str(acodec).lower():
                components.append("[opus]")
            else:
                components.append("[aac]")

        return " ".join(components)

    except Exception as e:
        logger.error(f"Error creating quality label: {str(e)}")
        return "Unknown Quality"


def consolidate_formats(formats: List[Dict], is_live: bool = False) -> List[FormatInfo]:
    """Enhanced format consolidation with better format grouping and display."""
    if is_live:
        return handle_live_formats(formats)

    if not formats:
        return [create_fallback_format()]

    # Resolution mapping
    RESOLUTION_MAP = {
        256: 240,
        426: 360,
        640: 360,
        854: 480,
        1280: 720,
        1920: 1080
    }

    try:
        # Separate formats by type
        combined_formats = []
        video_only_formats = []
        audio_formats = []

        for fmt in formats:
            try:
                if not fmt or fmt.get('ext') == 'mhtml':
                    continue

                raw_height = fmt.get('height', 0) or 0
                height = RESOLUTION_MAP.get(raw_height, raw_height)
                has_video = bool(fmt.get('vcodec') and fmt.get('vcodec') != 'none')
                has_audio = bool(fmt.get('acodec') and fmt.get('acodec') != 'none')
                vcodec = str(fmt.get('vcodec', '')).lower()

                # Skip VP9 formats
                if 'vp9' in vcodec or 'vp09' in vcodec:
                    continue

                # Skip formats without filesize unless necessary
                if not fmt.get('filesize') and 'storyboard' not in fmt.get('format_note', '').lower():
                    continue

                width = fmt.get('width', 0) or 0
                is_portrait = width < height if width and height else False
                fps = fmt.get('fps', 30) or 30

                # Enhanced codec detection
                if 'avc1' in vcodec or 'h264' in vcodec:
                    codec_family = 'h264'
                elif 'av01' in vcodec:
                    codec_family = 'av1'
                else:
                    continue

                # Categorize format
                if has_video and has_audio:
                    combined_formats.append((height, fps, codec_family, is_portrait, fmt))
                elif has_video and height >= 144:
                    video_only_formats.append((height, fps, codec_family, is_portrait, fmt))
                elif has_audio and not has_video:
                    audio_formats.append(fmt)

            except Exception as e:
                logger.error(f"Error processing format: {str(e)}")
                continue

        final_formats = []

        # Process combined formats first (h264 priority)
        h264_combined = [(h, f, c, p, fmt) for h, f, c, p, fmt in combined_formats if c == 'h264']
        other_combined = [(h, f, c, p, fmt) for h, f, c, p, fmt in combined_formats if c != 'h264']

        # Sort by height (desc), fps (desc)
        for height, fps, codec, is_portrait, fmt in sorted(h264_combined + other_combined,
                                                           key=lambda x: (-x[0], -x[1])):
            format_info = format_to_info({
                'format_id': fmt['format_id'],
                'ext': fmt.get('ext', 'mp4'),
                'quality': create_quality_label({
                    'height': height,
                    'width': fmt.get('width'),
                    'fps': fps,
                    'filesize': fmt.get('filesize'),
                    'vcodec': fmt.get('vcodec'),
                    'acodec': fmt.get('acodec'),
                    'is_portrait': is_portrait
                }),
                'filesize': fmt.get('filesize'),
                'vcodec': fmt.get('vcodec'),
                'acodec': fmt.get('acodec'),
                'width': fmt.get('width'),
                'height': height,
                'fps': fps,
                'tbr': fmt.get('tbr')
            })
            final_formats.append(format_info)

        # Process video-only formats
        h264_video = [(h, f, c, p, fmt) for h, f, c, p, fmt in video_only_formats if c == 'h264']
        av1_video = [(h, f, c, p, fmt) for h, f, c, p, fmt in video_only_formats if c == 'av1']

        for height, fps, codec, is_portrait, fmt in sorted(h264_video + av1_video,
                                                           key=lambda x: (-x[0], -x[1])):
            format_info = format_to_info({
                'format_id': fmt['format_id'],
                'ext': fmt.get('ext', 'mp4'),
                'quality': create_quality_label({
                    'height': height,
                    'width': fmt.get('width'),
                    'fps': fps,
                    'filesize': fmt.get('filesize'),
                    'vcodec': fmt.get('vcodec'),
                    'acodec': None,
                    'is_portrait': is_portrait
                }),
                'filesize': fmt.get('filesize'),
                'vcodec': fmt.get('vcodec'),
                'acodec': None,
                'width': fmt.get('width'),
                'height': height,
                'fps': fps,
                'tbr': fmt.get('tbr')
            })
            final_formats.append(format_info)

        # Process audio formats (limit to 128k)
        audio_formats.sort(key=lambda x: -(x.get('abr', 0) or x.get('tbr', 0) or 0))
        for fmt in audio_formats:
            if (fmt.get('abr', 0) or fmt.get('tbr', 0) or 0) <= 128:
                format_info = format_to_info({
                    'format_id': fmt['format_id'],
                    'ext': 'mp3',  # Will be converted to MP3
                    'quality': f"Audio {fmt.get('abr', fmt.get('tbr', 0))}k",
                    'filesize': fmt.get('filesize'),
                    'vcodec': None,
                    'acodec': fmt.get('acodec'),
                    'width': None,
                    'height': None,
                    'fps': None,
                    'tbr': fmt.get('tbr')
                })
                final_formats.append(format_info)

        # Set recommended format
        if final_formats:
            final_formats[0].is_recommended = True

        return final_formats

    except Exception as e:
        logger.error(f"Error in format consolidation: {str(e)}")
        return [create_fallback_format()]


def format_to_info(fmt: dict) -> FormatInfo:
    """Convert raw format dict to FormatInfo model with enhanced validation."""
    try:
        # Handle empty or invalid format
        if not fmt:
            return create_fallback_format()

        # Create display name and quality using unified function
        display_name = create_quality_label(fmt)
        fmt['display_name'] = display_name

        # Set quality string if not present
        if not fmt.get('quality'):
            fmt['quality'] = display_name

        # Create FormatInfo with proper field handling
        format_info = FormatInfo(
            format_id=fmt.get('format_id', 'best'),
            ext=fmt.get('ext', 'mp4'),
            quality=str(fmt.get('quality', '')),
            filesize=fmt.get('filesize'),
            vcodec=fmt.get('vcodec'),
            acodec=fmt.get('acodec'),
            width=fmt.get('width'),
            height=fmt.get('height'),
            fps=fmt.get('fps'),
            tbr=fmt.get('tbr'),
            compatibility_score=fmt.get('compatibility_score', 0),
            is_recommended=fmt.get('is_recommended', False),
            display_name=fmt.get('display_name', '')
        )

        return format_info

    except Exception as e:
        logger.error(f"Error converting format {fmt.get('format_id')}: {str(e)}")
        return create_fallback_format()


def create_fallback_format() -> FormatInfo:
    """Create a minimal valid format for error cases."""
    return FormatInfo(
        format_id='best',
        ext='mp4',
        quality='Default Quality',
        filesize=None,
        vcodec=None,
        acodec=None,
        width=None,
        height=None,
        fps=None,
        tbr=None,
        display_name='Default Quality',
        is_recommended=True
    )


def get_codec_type(codec: str) -> str:
    """Determine codec type from codec string."""
    codec = str(codec).lower()
    if 'opus' in codec:
        return 'opus'
    elif 'mp4a' in codec or 'aac' in codec:
        return 'aac'
    elif 'vorbis' in codec:
        return 'vorbis'
    return 'other'


def detect_hdr(fmt: Dict) -> bool:
    """Enhanced HDR detection with better null handling."""
    try:
        if not fmt:
            return False

        # Get format info with null safety
        vcodec = str(fmt.get('vcodec', '')).lower()
        quality = str(fmt.get('quality', '')).upper()
        height = fmt.get('height')

        # HDR indicators
        hdr_indicators = [
            'HDR' in quality,
            'BT2020' in str(fmt.get('color_info', '')).upper(),
            fmt.get('dynamic_range') == 'HDR',
            '.10.0.110' in vcodec,  # AV1 HDR indicator
            'VP9.2' in vcodec.upper(),  # VP9 HDR profile
            'AV1.2' in vcodec.upper(),  # AV1 HDR profile
            # Only check height if it's not None
            bool(height is not None and height >= 2160 and '.10.' in vcodec)
        ]

        return any(hdr_indicators)

    except Exception as e:
        logger.error(f"Error detecting HDR: {str(e)}")
        return False


def handle_live_formats(formats: List[Dict]) -> List[FormatInfo]:
    """Enhanced live stream format handling with adaptive bitrates."""
    try:
        live_formats = []

        # Define bitrate tiers
        bitrate_tiers = [
            ('Low', 1000),  # 1 Mbps
            ('Medium', 2500),  # 2.5 Mbps
            ('High', 5000),  # 5 Mbps
            ('Ultra', 8000)  # 8 Mbps
        ]

        # Process existing formats
        valid_formats = [f for f in formats if f.get('vcodec')
                         and f.get('acodec')
                         and f.get('tbr', 0) < 10000]  # Cap at 10 Mbps for stability

        if valid_formats:
            # Sort by bitrate
            valid_formats.sort(key=lambda x: x.get('tbr', 0))

            # Select representative qualities
            for quality, target_bitrate in bitrate_tiers:
                closest = min(valid_formats,
                              key=lambda x: abs((x.get('tbr', 0) or 0) - target_bitrate),
                              default=None)

                if closest:
                    fmt = format_to_info({
                        'format_id': f'live-{quality.lower()}',
                        'ext': 'mp4',
                        'quality': create_quality_label({
                            'height': closest.get('height'),
                            'width': closest.get('width'),
                            'fps': closest.get('fps'),
                            'vcodec': closest.get('vcodec'),
                            'acodec': closest.get('acodec'),
                            'tbr': closest.get('tbr'),
                            'is_live': True
                        }),
                        'filesize': None,
                        'vcodec': closest.get('vcodec'),
                        'acodec': closest.get('acodec'),
                        'width': closest.get('width'),
                        'height': closest.get('height'),
                        'fps': closest.get('fps'),
                        'tbr': closest.get('tbr'),
                        'display_name': (f"{quality} Quality - "
                                         f"{closest.get('height', 'Auto')}p "
                                         f"({closest.get('tbr', 0) / 1000:.1f} Mbps)")
                    })
                    live_formats.append(fmt)

        # Always add adaptive option
        live_formats.append(FormatInfo(
            format_id='best',
            ext='mp4',
            quality='Adaptive Quality (Live)',
            filesize=None,
            vcodec=None,
            acodec=None,
            width=None,
            height=None,
            fps=None,
            tbr=None,
            display_name='Live Stream (Adaptive Quality)',
            is_recommended=True
        ))

        return live_formats

    except Exception as e:
        logger.error(f"Error handling live formats: {str(e)}")
        return [FormatInfo(
            format_id='best',
            ext='mp4',
            quality='Adaptive Quality (Live)',
            filesize=None,
            vcodec=None,
            acodec=None,
            width=None,
            height=None,
            fps=None,
            tbr=None,
            display_name='Live Stream (Adaptive Quality)',
            is_recommended=True,
            error=str(e)
        )]


def sanitize_filename(title: str) -> str:
    """Sanitize the filename to remove invalid characters and ensure proper encoding."""
    # Remove invalid filename characters
    title = re.sub(r'[<>:"/\\|?*]', '', title)
    # Replace multiple spaces with single space
    title = re.sub(r'\s+', ' ', title).strip()
    # Encode and decode to handle any Unicode characters
    try:
        title.encode('ascii')
        return title[:200]
    except UnicodeEncodeError:
        return unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('ascii')[:200]


def format_size(size_in_bytes: Optional[int]) -> str:
    """Convert bytes to human readable string with enhanced precision."""
    try:
        if size_in_bytes is None:
            return "Size unknown"

        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_in_bytes < 1024:
                if unit == 'B':
                    return f"{size_in_bytes:.0f} {unit}"
                return f"{size_in_bytes:.1f} {unit}"
            size_in_bytes /= 1024

        return f"{size_in_bytes:.1f} PB"

    except Exception as e:
        logger.error(f"Error formatting size: {str(e)}")
        return "Size unknown"


def extract_format_info(format_data: Dict) -> Dict:
    """Extract and validate format information with enhanced error checking."""
    try:
        return {
            'format_id': format_data.get('format_id', 'unknown'),
            'ext': format_data.get('ext', 'mp4'),
            'quality': str(format_data.get('quality', '')),
            'filesize': format_data.get('filesize'),
            'vcodec': format_data.get('vcodec'),
            'acodec': format_data.get('acodec'),
            'width': format_data.get('width'),
            'height': format_data.get('height'),
            'fps': format_data.get('fps'),
            'tbr': format_data.get('tbr'),
            'format_note': format_data.get('format_note', ''),
            'dynamic_range': format_data.get('dynamic_range', '')
        }

    except Exception as e:
        logger.error(f"Error extracting format info: {str(e)}")
        return create_fallback_format().__dict__


def detect_device_type(
        user_agent: Optional[str] = Header(None),
        requested_type: Optional[DeviceType] = None
) -> DeviceType:
    """Smart device type detection with multiple fallback strategies."""
    # Honor explicit user preference if provided
    if requested_type:
        return requested_type

    # Fallback to user agent detection if available
    if user_agent:
        try:
            ua = parse(user_agent)
            return DeviceType.MOBILE if ua.is_mobile else DeviceType.DESKTOP
        except Exception as e:
            logger.warning(f"Error parsing user agent: {e}")

    # Default to desktop as safest fallback
    return DeviceType.DESKTOP


def is_format_compatible(format_data: Dict, device_type: DeviceType) -> bool:
    """Check if a format is compatible with the device type."""
    constraints = DEVICE_CONSTRAINTS[device_type]

    # Skip height check for audio-only formats
    if format_data.get('vcodec') != 'none':
        height = format_data.get('height')
        if height is not None and height > constraints['max_height']:
            return False

        fps = format_data.get('fps')
        if fps is not None and round(fps) > constraints['max_fps']:  # Round FPS for comparison
            return False

        if (format_data.get('vcodec') and
                not any(codec in format_data.get('vcodec', '')
                        for codec in constraints['preferred_codecs'])):
            return False

    if (constraints['max_size_mb'] and format_data.get('filesize')):
        size_mb = format_data['filesize'] / (1024 * 1024)
        if size_mb > constraints['max_size_mb']:
            return False

    return True


def calculate_compatibility_score(format_data: Dict, device_type: DeviceType) -> int:
    """Calculate format compatibility score with enhanced criteria."""
    if not format_data or not isinstance(format_data, dict):
        return 0

    try:
        constraints = DEVICE_CONSTRAINTS[device_type]
        score = 0

        # Handle live streams differently
        if format_data.get('is_live', False):
            return calculate_live_stream_score(format_data, device_type)

        # Resolution score (max 40 points)
        height = format_data.get('height', 0) or 0
        if height > 0:
            height_score = min(40, int((height / constraints['max_height']) * 40))
            score += height_score

        # Codec compatibility (max 30 points)
        vcodec = str(format_data.get('vcodec', '')).lower()
        if any(codec in vcodec for codec in constraints['preferred_codecs']):
            score += 30
            # Bonus for most compatible codec
            if 'avc1' in vcodec or 'h264' in vcodec:
                score += 5

        # FPS compatibility (max 15 points)
        fps = format_data.get('fps', 0) or 0
        if fps > 0:
            if fps <= constraints['max_fps']:
                score += 15
            else:
                score += max(0, 15 - (fps - constraints['max_fps']))

        # File size consideration (max 10 points)
        if constraints['max_size_mb'] and format_data.get('filesize'):
            size_mb = format_data['filesize'] / (1024 * 1024)
            if size_mb <= constraints['max_size_mb']:
                size_score = (1 - (size_mb / constraints['max_size_mb'])) * 10
                score += size_score

        # Bitrate consideration (max 5 points)
        if constraints['max_bitrate'] and format_data.get('tbr'):
            tbr = format_data['tbr']
            if tbr <= constraints['max_bitrate']:
                bitrate_score = (1 - (tbr / constraints['max_bitrate'])) * 5
                score += bitrate_score

        return int(min(100, score))

    except Exception as e:
        logger.error(f"Error calculating compatibility score: {str(e)}")
        return 0


def calculate_live_stream_score(format_data: Dict, device_type: DeviceType) -> int:
    """Calculate compatibility score specifically for live streams."""
    try:
        constraints = DEVICE_CONSTRAINTS[device_type]
        score = 50  # Base score for live streams

        # Bitrate scoring
        tbr = format_data.get('tbr', 0) or 0
        if tbr > 0:
            if constraints['max_bitrate']:
                if tbr <= constraints['max_bitrate']:
                    score += 30
                else:
                    score -= min(30, int((tbr - constraints['max_bitrate']) / 1000))

        # Resolution scoring
        height = format_data.get('height', 0) or 0
        if height > 0:
            if height <= constraints['max_height']:
                score += 20
            else:
                score -= min(20, int((height - constraints['max_height']) / 100))

        return int(max(0, min(100, score)))

    except Exception as e:
        logger.error(f"Error calculating live stream score: {str(e)}")
        return 0


async def analyze_media(url: str, device_type: DeviceType) -> MediaAnalysis:
    """Enhanced media analysis with better format extraction."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'youtube_include_dash_manifest': True,
        'extract_flat': False,
        'format': None,  # Don't specify format to get all available formats
        'extractor_args': {
            'youtube': {
                'formats': 'incomplete',  # New way to include all formats
            }
        }
    }

    # Special handling for Instagram
    if 'instagram.com' in url:
        ydl_opts.update({
            'extract_flat': False,
            'extractor_args': {
                'instagram': {
                    'formats': 'incomplete',
                }
            }
        })

    try:
        # Check if URL is supported
        extractors = yt_dlp.extractor.gen_extractors()
        if not any(e.suitable(url) for e in extractors):
            return create_empty_response(url)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # Debug logging to see all available formats
            logger.debug(f"Available formats for {url}:")
            for fmt in info.get('formats', []):
                logger.debug(
                    f"Format: {fmt.get('format_id')} - "
                    f"res: {fmt.get('height')}p - "
                    f"vcodec: {fmt.get('vcodec')} - "
                    f"acodec: {fmt.get('acodec')} - "
                    f"fps: {fmt.get('fps')} - "
                    f"HDR: {'HDR' in str(fmt.get('format_note', '')).upper()} - "
                    f"size: {fmt.get('filesize')} - "
                    f"tbr: {fmt.get('tbr')} - "
                    f"dynamic_range: {fmt.get('dynamic_range')} - "
                    f"color_info: {fmt.get('color_info')} - "
                    f"format_note: {fmt.get('format_note')}"
                )

            is_live = info.get('is_live', False)
            formats = info.get('formats', [])

            # Filter out formats with no video/audio codecs
            formats = [f for f in formats if f.get('vcodec') != 'none' or f.get('acodec') != 'none']

            # Add HDR detection
            for fmt in formats:
                hdr_indicators = [
                    'HDR' in str(fmt.get('format_note', '')).upper(),
                    'BT2020' in str(fmt.get('color_info', '')).upper(),
                    fmt.get('dynamic_range') == 'HDR',
                    'VP9.2' in str(fmt.get('vcodec', '')).upper(),
                    'AV1.2' in str(fmt.get('vcodec', '')).upper(),
                ]
                if any(hdr_indicators):
                    fmt['dynamic_range'] = 'HDR'

            consolidated_formats = consolidate_formats(formats, is_live)

            return MediaAnalysis(
                url=url,
                title=info.get('title', ''),
                downloadable=True,
                formats=consolidated_formats,
                is_live=is_live,
                duration=info.get('duration'),
                thumbnail=info.get('thumbnail')
            )

    except Exception as e:
        logger.error(f"Error analyzing media: {str(e)}", exc_info=True)
        return create_empty_response(url)


def create_empty_response(url: str) -> MediaAnalysis:
    """Create an empty response for unsupported or error cases."""
    return MediaAnalysis(
        url=url,
        title="Error processing URL",
        downloadable=False,
        formats=[],
        is_live=False,
        duration=None,
        thumbnail=None
    )


async def download_file(url: str, format_id: str, temp_dir: str,
                        embed_thumbnail: bool = True, duration: Optional[int] = None) -> Tuple[str, str]:
    """Download file in a separate thread with automatic format handling."""
    try:
        # Get video info first to get the title and check format
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            safe_title = sanitize_filename(info.get('title', ''))
            is_live = info.get('is_live', False)
            formats = info.get('formats', [])

        # Set default duration for live streams if not provided
        if is_live and duration is None:
            duration = 15

        # Find selected format
        selected_format = next((f for f in formats if f.get('format_id') == format_id), None)
        if selected_format:
            has_video = selected_format.get('vcodec') and selected_format.get('vcodec') != 'none'
            has_audio = selected_format.get('acodec') and selected_format.get('acodec') != 'none'

            # If video-only, find best audio <= 128k
            if has_video and not has_audio:
                audio_formats = [f for f in formats
                                 if f.get('acodec') and f.get('acodec') != 'none'
                                 and not (f.get('vcodec') and f.get('vcodec') != 'none')
                                 and ((f.get('abr', 0) or 0) <= 128
                                      or (f.get('tbr', 0) or 0) <= 128)]

                if audio_formats:
                    # Sort by quality and prefer certain codecs
                    best_audio = max(audio_formats,
                                     key=lambda x: (3 if 'opus' in str(x.get('acodec', '')).lower()
                                                    else 2 if 'aac' in str(x.get('acodec', '')).lower()
                                     else 1,
                                                    x.get('abr', 0) or x.get('tbr', 0) or 0))
                    format_id = f"{format_id}+{best_audio['format_id']}"

            # If audio-only, set up for MP3 conversion
            elif not has_video and has_audio:
                postprocessors = [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '128',
                }]

                if embed_thumbnail and info.get('thumbnail'):
                    postprocessors.append({
                        'key': 'EmbedThumbnail',
                        'already_have_thumbnail': False,
                    })

                ydl_opts = {
                    'format': format_id,
                    'outtmpl': str(Path(temp_dir) / f'{safe_title}.%(ext)s'),
                    'postprocessors': postprocessors,
                    'writethumbnail': embed_thumbnail,
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    logger.info(f"Starting audio download for format {format_id}")
                    download_info = ydl.extract_info(url, download=True)
                    filename = ydl.prepare_filename(download_info)
                    mp3_filename = str(Path(filename).with_suffix('.mp3'))
                    return mp3_filename, f"{safe_title}.mp3"

        # Configure postprocessors for video
        postprocessors = [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }]

        if embed_thumbnail and info.get('thumbnail'):
            postprocessors.append({
                'key': 'EmbedThumbnail',
                'already_have_thumbnail': False,
            })

        # Configure download options
        if is_live and duration:
            ydl_opts = {
                'format': 'best',
                'outtmpl': str(Path(temp_dir) / f'{safe_title}.%(ext)s'),
                'external_downloader': 'ffmpeg',
                'external_downloader_args': {
                    'ffmpeg_i': ['-t', str(duration)]
                },
                'postprocessors': postprocessors,
                'writethumbnail': embed_thumbnail,
            }
        else:
            ydl_opts = {
                'format': format_id,
                'outtmpl': str(Path(temp_dir) / f'{safe_title}.%(ext)s'),
                'merge_output_format': 'mp4',
                'postprocessors': postprocessors,
                'writethumbnail': embed_thumbnail,
            }

        # Download the file
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


@app.post("/analyze", response_model=List[MediaAnalysis])
async def analyze_urls(
        request: DownloadRequest,
        user_agent: Optional[str] = Header(None)
):
    """Analyze URLs and return compatible formats."""
    device_type = detect_device_type(user_agent, request.device_type)
    tasks = [analyze_media(url, device_type) for url in request.urls]
    return await asyncio.gather(*tasks)


@app.post("/download")
async def download_media_endpoint(
        download_request: DownloadFormat,
        background_tasks: BackgroundTasks,
        user_agent: Optional[str] = Header(None)
):
    """Download media in specified format."""
    device_type = detect_device_type(user_agent, download_request.device_type)

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # First verify the format is compatible
        analysis = await analyze_media(download_request.url, device_type)
        if not analysis.downloadable:
            raise HTTPException(status_code=400, detail="URL is not supported for download")

        if not any(f.format_id == download_request.format_id for f in analysis.formats):
            raise HTTPException(status_code=400, detail="Requested format is not compatible with device")

        # Download the file
        filename, output_filename = await asyncio.get_event_loop().run_in_executor(
            download_executor,
            lambda: asyncio.run(download_file(
                download_request.url,
                download_request.format_id,
                temp_dir,
                download_request.embed_thumbnail,
                download_request.duration
            ))
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

        # Get format info for content type
        format_info = next(f for f in analysis.formats if f.format_id == download_request.format_id)
        content_type = f"video/{format_info.ext}" if format_info.vcodec else f"audio/{format_info.ext}"

        return Response(
            content=file_content,
            media_type=content_type,
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
