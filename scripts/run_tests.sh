#!/usr/bin/env bash
# Single entry point for the whole unit test suite (ingestion, modeling,
# logging, API). Every test in tests/ is isolated from the real
# models/ and logs/ directories via tmp_path + monkeypatch, so running
# this is always safe to run against a live/production checkout.
#
# Usage: ./scripts/run_tests.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pytest tests/ -v
