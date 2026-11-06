set windows-shell := ["pwsh.exe", "-c"]

# Print the help message.
@help:
    echo "Usage: just [RECIPE]\n"
    just --list

# Build documentation.
docs:
    uv run zensical build --strict

# Serve documentation locally for development.
serve-docs:
    uv run zensical serve

# Clean documentation build artifacts.
[unix]
clean-docs:
    rm -rf .cache/ site/

# Clean documentation build artifacts.
[windows]
clean-docs:
    foreach ($p in '.cache', 'site') { if (Test-Path $p) { Remove-Item -Recurse -Force $p } }

# Run all tests.
test:
    uv run pytest tests/

# Run tests with verbose output.
test-verbose:
    uv run pytest tests/ -v

# Run all pre-commit hooks.
pre-commit:
    uv run prek run --all-files

# Create or reuse the isolated Python 3.13 environment with qualified ANTsPy.
[unix]
antspy-venv:
    test -x .venv-antspy/bin/python || uv venv --python 3.13 .venv-antspy
    uv pip install --python .venv-antspy/bin/python 'antspyx==0.6.3'

# Benchmark resampling into a fresh output directory; extra arguments go to Python.
[unix]
[positional-arguments]
benchmark-resample output *args: antspy-venv
    uv run --extra examples python examples/benchmark_resample.py --antspy-python "{{justfile_directory()}}/.venv-antspy/bin/python" --output "$@"

# Benchmark registration into a fresh output directory; extra arguments go to Python.
[unix]
[positional-arguments]
benchmark-register output *args: antspy-venv
    uv run --extra examples python examples/benchmark_register.py --antspy-python "{{justfile_directory()}}/.venv-antspy/bin/python" --output "$@"

# Render one or more benchmark run directories or results.json files as HTML.
[unix]
[positional-arguments]
benchmark-report output +results:
    uv run --extra examples python examples/benchmark_report.py --output "$@"

# Aliases
alias d := docs
alias cd := clean-docs
alias sd := serve-docs
alias t := test
alias tv := test-verbose
alias pc := pre-commit
