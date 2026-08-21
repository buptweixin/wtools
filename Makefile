.PHONY: format style_check test lint install dev-install clean

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"
	pip install -r requirements-dev.txt

format:
	black .
	isort .

style_check:
	isort --diff --check .
	black --check --diff .

test:
	pytest

lint:
	mypy wtools tools
	black --check .
	isort --check .

clean:
	rm -rf build/ dist/ *.egg-info wtools.egg-info/ .eggs/
