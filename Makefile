test:
	pytest;
	pydocstyle --convention=google condense tests;
	flake8 . --per-file-ignores="__init__.py:F401" --select=E9,F63,F7,F82 --show-source --statistics;
	flake8 . --per-file-ignores="__init__.py:F401" --exit-zero --max-complexity=10 --max-line-length=120 --statistics;
