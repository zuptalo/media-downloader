# Media Download API

[![CI/CD Pipeline](https://github.com/zuptalo/media-downloader/actions/workflows/ci.yaml/badge.svg)](https://github.com/zuptalo/media-downloader/actions/workflows/ci.yaml)
[![Docker Image Version (latest semver)](https://img.shields.io/docker/v/zuptalo/media-downloader?sort=semver&logo=docker)](https://hub.docker.com/r/zuptalo/media-downloader)
[![Docker Pulls](https://img.shields.io/docker/pulls/zuptalo/media-downloader)](https://hub.docker.com/r/zuptalo/media-downloader)
[![codecov](https://codecov.io/gh/zuptalo/media-downloader/branch/main/graph/badge.svg)](https://codecov.io/gh/zuptalo/media-downloader)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A containerized FastAPI service that provides video/audio downloading capabilities using yt-dlp. Built for easy
integration with iOS Shortcuts or any other client application.

## Features

- 🔍 URL Analysis: Get available formats, qualities, and metadata
- 📥 Flexible Downloads: Support for various formats and quality options
- 🔄 Smart Processing: Automatic video/audio stream merging for optimal quality
- 🖼️ Thumbnail Integration: Optional thumbnail embedding in downloads
- 📺 Live Stream Support: Download live content with duration control
- ⚡ High Performance:
    - Parallel download processing
    - Asynchronous request handling
    - Efficient temporary file management
- 🐳 Production Ready:
    - Full Docker support
    - Kubernetes deployment configurations
    - Health checks and monitoring

## Prerequisites

### Development Dependencies

- Python 3.11+
- FFmpeg
- Docker (optional, for containerization)
- Kubernetes (optional, for production deployment)

### Quick Install Guide

#### macOS

```bash
# Install using Homebrew
brew install python@3.11 ffmpeg

# If you don't have Homebrew:
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Ubuntu/Debian

```bash
sudo apt update
sudo apt install python3.11 ffmpeg
```

#### Windows

- Install Python 3.11+ from [python.org](https://www.python.org/downloads/)
- Install FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)

### Core Dependencies

```
fastapi==0.109.1
uvicorn==0.24.0
yt-dlp==2024.7.7
pydantic==2.5.1
python-multipart==0.0.7
```

## Getting Started

### Local Development

1. Create and activate virtual environment:
    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    python app/main.py
    ```

The API will be available at `http://localhost:8000`

### Docker Deployment

1. Build the image:
    ```bash
    docker build -t media-downloader .
    ```

2. Run the container:
    ```bash
    docker run -p 8000:8000 media-downloader
    ```

## Required Repository Secrets

Before deploying, ensure you have set up the following secrets in your GitHub repository:

### Authentication Secrets

- `GITHUB_TOKEN`: Automatically provided by GitHub Actions
- `DOCKERHUB_USERNAME`: Your Docker Hub username
- `DOCKERHUB_TOKEN`: Your Docker Hub access token
- `CODECOV_TOKEN`: Token for CodeCov integration

### Kubernetes Deployment Secrets

- `KUBE_CONFIG`: Your base64-encoded kubeconfig file
- `DOMAIN_NAME`: The domain name for your deployment (optional, falls back to default in workflow)

To add these secrets:

1. Go to your repository settings
2. Navigate to Secrets and Variables > Actions
3. Click "New repository secret"
4. Add each secret with its corresponding value

## Kubernetes Deployment

The deployment process has been streamlined to generate Kubernetes manifests during the deployment workflow. This
approach provides:

- Dynamic configuration based on environment variables
- Automatic version management
- Simplified maintenance

To deploy:

1. Ensure you have set up all required secrets mentioned above

2. The deployment workflow will automatically:
    - Generate necessary Kubernetes manifests
    - Create/update ConfigMaps
    - Deploy the application with proper configuration
    - Set up ingress with TLS
    - Configure autoscaling

3. Monitor the deployment:
   ```bash
   kubectl get pods -n media-downloader
   kubectl get ingress -n media-downloader
   kubectl get services -n media-downloader
   ```

## API Documentation

### Analyze URLs

`POST /analyze`

Get detailed information about media URLs including available formats and metadata. Supports analyzing multiple URLs in
a single request.

#### Request

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://www.youtube.com/watch?v=example1",
      "https://www.youtube.com/watch?v=example2"
    ]
  }'
```

#### Response

```json
[
  {
    "url": "string",
    "title": "string",
    "formats": [
      {
        "format_id": "string",
        "ext": "string",
        "quality": "string | integer | float | null",
        "filesize": "integer | null",
        "format_note": "string | null",
        "acodec": "string | null",
        "vcodec": "string | null"
      }
    ],
    "thumbnail": "string | null",
    "channel": "string | null",
    "upload_date": "string | null",
    "description": "string | null",
    "is_live": "boolean"
  }
]
```

The response is an array of media information objects, one for each URL provided in the request. Each object contains:

- `url`: The original URL requested
- `title`: Title of the media
- `formats`: Array of available format options
    - `format_id`: Unique identifier for the format (used in download requests)
    - `ext`: File extension
    - `quality`: Quality indicator (when available)
    - `filesize`: Size in bytes (when available)
    - `format_note`: Additional format information
    - `acodec`: Audio codec
    - `vcodec`: Video codec
- `thumbnail`: URL of the media thumbnail
- `channel`: Channel or uploader name
- `upload_date`: Date the media was uploaded
- `description`: Media description
- `is_live`: Boolean indicating if this is a live stream

### Download Media

`POST /download`

Download media in specified format with optional features.

```bash
curl -X POST http://localhost:8000/download \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.youtube.com/watch?v=example",
    "format_id": "137",
    "embed_thumbnail": true
  }' \
  --remote-header-name --remote-name
```

#### Parameters

- `url`: Media URL
- `format_id`: Format identifier (from analyze response)
- `embed_thumbnail`: Include video thumbnail (default: true)
- `duration`: Duration limit in seconds (for live streams)

### Live Stream Downloads

```bash
curl -X POST http://localhost:8000/download \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.youtube.com/watch?v=live-example",
    "format_id": "best",
    "duration": 30
  }' \
  --remote-header-name --remote-name
```

## iOS Shortcuts Integration

### Basic Setup

1. Create new shortcut
2. Add "URL" action with your API endpoint
3. Configure "Get Contents of URL" action

### Example Workflows

#### Format Analyzer

```
1. URL → API_ENDPOINT/analyze
2. Get Contents of URL
   - Method: POST
   - Headers: Content-Type: application/json
   - Body: {"urls": ["<clipboard>"]}
3. Parse JSON Response
4. Show Format Options
```

#### Video Downloader

```
1. URL → API_ENDPOINT/download
2. Get Contents of URL
   - Method: POST
   - Headers: Content-Type: application/json
   - Body: {
     "url": "<video URL>",
     "format_id": "<selected format>",
     "embed_thumbnail": true
   }
3. Save File
```

## Security Considerations

- Implement authentication for production use
- Configure rate limiting
- Monitor storage usage
- Use HTTPS in production
- Validate user input
- Regular dependency updates

## Error Handling

The API provides comprehensive error handling:

- Invalid URL/format validation
- Download failure recovery
- Live stream processing errors
- Resource management
- Network issues

All errors return appropriate HTTP status codes with detailed messages.

## CI/CD

This project uses GitHub Actions for continuous integration and deployment:

### Container Registries

Images are automatically built and pushed to:

- GitHub Container Registry (GHCR): `ghcr.io/zuptalo/media-downloader`
- Docker Hub: `zuptalo/media-downloader`

### Automated Builds

- Builds trigger on:
    - Pushes to main branch
    - Pull requests to main
    - Tag creation (v*.*.*)
- Multi-platform builds: linux/amd64, linux/arm64
- Includes security signing for GHCR images
- Automated version tagging based on git tags

### Using Pre-built Images

```bash
# Pull from GHCR
docker pull ghcr.io/zuptalo/media-downloader:latest

# Pull from Docker Hub
docker pull zuptalo/media-downloader:latest
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your PR adheres to:

- Consistent code style
- Proper documentation
- Test coverage
- Clear commit messages

## License

MIT License

Copyright (c) 2024 Zuptalo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.