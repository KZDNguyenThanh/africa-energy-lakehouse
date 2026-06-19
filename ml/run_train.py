"""Local entry point for the ML stage.

Thin wrapper around ml/benchmark_models.py so that the multi-algorithm benchmark,
evaluation, tuning, SHAP, and MotherDuck export live in a single source of truth.
Run with the project .venv: python ml/run_train.py
"""
from benchmark_models import main

if __name__ == "__main__":
    main()
