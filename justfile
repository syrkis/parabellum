# Default recipe to run when just is called without arguments
default:
    @just --list

# Run the Python script with Poetry
run *ARGS:
    time poetry run python ludens.py {{ARGS}}

# Install dependencies
install:
    poetry install

# Update dependencies
update:
    poetry update
