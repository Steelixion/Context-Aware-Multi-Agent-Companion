#!/usr/bin/env bash
# Starts the backend with camera, mic, local LLM and Piper voice enabled.
cd "$(dirname "$0")"
export DC_VISION_MODE=camera DC_ENABLE_MIC=1
exec .venv311/bin/python -m app.main
