.PHONY: install test lint headline sweep figures all clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check anfis experiments tests scripts

headline:
	python -m experiments.run_headline --dataset mackey_glass --seeds 5
	python -m experiments.run_headline --dataset nonlinear_mixed --seeds 5

sweep:
	python -m experiments.run_sweep --dataset mackey_glass --folds 5

figures:
	python -m experiments.make_figures --dataset mackey_glass

all: test headline sweep figures

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache *.egg-info build dist
