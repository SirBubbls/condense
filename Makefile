test: pytest style_checks

pytest:
	pytest --cov=condense tests

style_checks: docstyle
	flake8 condense --per-file-ignores="__init__.py:F401" --select=E9,F63,F7,F82 --show-source --statistics;
	flake8 condense --per-file-ignores="__init__.py:F401" --exit-zero --max-complexity=10 --max-line-length=120 --statistics;
	flake8 tests --per-file-ignores="__init__.py:F401" --select=E9,F63,F7,F82 --show-source --statistics;
	flake8 tests --per-file-ignores="__init__.py:F401" --exit-zero --max-complexity=10 --max-line-length=120 --statistics;

docstyle:
	pydocstyle --convention=google condense tests;
