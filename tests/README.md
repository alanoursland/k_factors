
# Tests

This project uses **pytest**. With a local Python env (torch/numpy installed), run:

```bash
pip install -r requirements.txt  # if present
pip install pytest coverage      # dev tools
pytest
```

The test suite assumes the source lives in `src/` (added to `sys.path` via `tests/conftest.py`).
Optional tools configured in `pyproject.toml`: pytest and coverage.
