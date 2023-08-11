style:
	black --line-length 119 --target-version py311 .
	isort --skip .git --skip .venv .

quality:
	black --check --line-length 119 --target-version py311 .
	isort --skip .git --skip .venv --check-only .
	flake8 . --exclude __pycache__,__init__.py,.venv/,.git/

