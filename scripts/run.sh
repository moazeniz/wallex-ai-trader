#!/bin/bash
#
# CryptoAlpha Run Script
#
# Usage:
#   ./scripts/run.sh              # Run full system
#   ./scripts/run.sh analyze BTC  # Quick analysis
#   ./scripts/run.sh bot          # Telegram bot only
#   ./scripts/run.sh performance  # Show performance
#

set -e

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
elif [ -f "test.env" ]; then
    export $(cat test.env | grep -v '^#' | xargs)
fi

# Check Python
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

# Run the application
case "${1:-run}" in
    run)
        echo "Starting CryptoAlpha system..."
        python main.py run "${@:2}"
        ;;
    analyze)
        if [ -z "$2" ]; then
            echo "Usage: $0 analyze SYMBOL [--timeframe 1h]"
            exit 1
        fi
        python main.py analyze "$2" "${@:3}"
        ;;
    bot)
        echo "Starting Telegram bot..."
        python main.py bot "${@:2}"
        ;;
    performance)
        python main.py performance "${@:2}"
        ;;
    test)
        echo "Running tests..."
        python -m pytest tests/ -v
        ;;
    *)
        echo "Unknown command: $1"
        echo "Available commands: run, analyze, bot, performance, test"
        exit 1
        ;;
esac
