#!/bin/bash

# Usage: ./run_docker.sh [openrouter|claude|gemini|openai]
# Example: ./run_docker.sh claude

set -e

PROVIDER=${1:-openrouter}

case "$PROVIDER" in
    openrouter|claude|gemini|openai) ;;
    *)
        echo "Error: Provider must be 'openrouter', 'claude', 'gemini' or 'openai'"
        echo "Usage: ./run_docker.sh [openrouter|claude|gemini|openai]"
        exit 1
        ;;
esac

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please create one from .env.example"
    exit 1
fi

# Build the docker run command
DOCKER_CMD="docker run -it --env-file .env"

echo "Running with $PROVIDER configuration..."

# Mount the codex config for the worker (sub-agent) if present. Any non-OpenAI
# worker provider (OpenRouter, Claude, Gemini, ...) needs this so the codex binary knows
# which endpoint to talk to.
if [ -f "$HOME/.codex/config.toml" ]; then
    echo "Mounting $HOME/.codex/config.toml for the codex worker"
    DOCKER_CMD="$DOCKER_CMD -v $HOME/.codex/config.toml:/root/.codex/config.toml:ro"
elif [ "$PROVIDER" != "openai" ]; then
    echo "Error: $HOME/.codex/config.toml not found (required so the codex worker uses $PROVIDER)"
    exit 1
fi

# Add volume mount for logs
DOCKER_CMD="$DOCKER_CMD -v $(pwd)/logs:/app/trinity/ARTEMIS/logs"

# Run the container
$DOCKER_CMD artemis \
    python -m supervisor.supervisor \
      --config-file configs/tests/ctf_easy.yaml \
      --benchmark-mode \
      --duration 10 \
      --skip-todos
