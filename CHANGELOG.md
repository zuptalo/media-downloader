
## [0.0.1] - 2024-12-06
### Changes
- Fixing auto-version.yaml so it happens only once after both docker build and pushes
- Adding auto-version.yaml, release.yaml and CHANGELOG.md
- Fixing docker-build.yaml
- Initial Commit

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project setup
- FastAPI application for media downloads
- Docker container support
- Kubernetes deployment manifests
- GitHub Actions for CI/CD
- Support for multiple video/audio formats
- Automatic thumbnail embedding
- Live stream support
- Health checks and monitoring

### Changed

### Deprecated

### Removed

### Fixed

### Security
- Implemented security context in Kubernetes deployment
- Added resource limits and requests
- Configured non-root user execution

## [1.0.0] - 2024-12-06
### Added
- Initial release