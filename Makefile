.PHONY: help install test run synthetic clean lint

help:
	@echo "install    pip install -r requirements.txt"
	@echo "test       run the 20-case pytest suite (no network)"
	@echo "synthetic  full pipeline on generated data (no FRED key)"
	@echo "run        full pipeline on real FRED data (needs FRED_API_KEY)"
	@echo "clean      remove caches, figures and the local database"

install:
	pip install -r requirements.txt

test:
	pytest -q

synthetic:
	python -m src.run_all --synthetic

run:
	@test -n "$$FRED_API_KEY" || (echo "ERROR: export FRED_API_KEY first" && exit 1)
	python -m src.run_all

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache figures/*.png figures/*.csv data/*.db