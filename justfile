set windows-shell := ["pwsh.exe", "-c"]

# Print the help message.
@help:
    echo "Usage: just [RECIPE]\n"
    just --list

# Run all tests.
test:
    uv run pytest tests/

# Run tests with verbose output.
test-verbose:
    uv run pytest tests/ -v

# Run all pre-commit hooks.
pre-commit:
    uv run prek run --all-files

# Benchmark registration; add --lazy for streaming read/fit/Zarr writes.
[unix]
[positional-arguments]
benchmark-register output *args:
    @just _benchmark register "$@"

# Benchmark resampling; add --lazy for streaming read/sample/Zarr writes.
[unix]
[positional-arguments]
benchmark-resample output *args:
    @just _benchmark resample "$@"

[private]
[unix]
[positional-arguments]
_benchmark operation output *args:
    #!/usr/bin/env bash
    set -euo pipefail
    operation="$1"; shift
    options=()
    if [[ " $* " != *" --lazy "* && " $* " != *" --list "* && " $* " != *" --help "* ]]; then
        just antspy-venv
        options=(--antspy-python "{{justfile_directory()}}/.venv-antspy/bin/python")
    fi
    uv run --extra benchmarks python "benchmarks/benchmark_${operation}.py" "${options[@]}" --output "$@"

# Create or reuse the isolated Python 3.13 environment with qualified ANTsPy.
[private]
[unix]
antspy-venv:
    test -x .venv-antspy/bin/python || uv venv --python 3.13 .venv-antspy
    uv pip install --python .venv-antspy/bin/python 'antspyx==0.6.3'

# Render eager benchmark results as HTML; lazy runs include their own report.
[unix]
[positional-arguments]
benchmark-report output +results:
    uv run --extra benchmarks python benchmarks/benchmark_report.py --output "$@"

# Aliases
alias t := test
alias tv := test-verbose
alias pc := pre-commit
