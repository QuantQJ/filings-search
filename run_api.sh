#!/bin/bash
cd "$(dirname "$0")"
exec ./.venv/bin/uvicorn filings_search.api:app --host 127.0.0.1 --port 8801 "$@"
