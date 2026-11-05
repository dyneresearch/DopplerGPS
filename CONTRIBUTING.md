# Contributing to DopplerGPS

Thank you for your interest in contributing to **DopplerGPS**, developed by [Dyne Research](https://dyneresearch.com). We welcome contributions from researchers, geospatial engineers, acoustics specialists, and open-source intelligence analysts.

---

## 🔬 Scientific & Ethical Standards

DopplerGPS is an open-source scientific instrument. All contributors must adhere to ethical research standards:
- **Responsible Use:** Contributions intended to enable non-consensual surveillance, stalking, unlawful tracking, or kinetic targeting will be rejected immediately.
- **Data Integrity:** Synthetic and benchmark test cases must be mathematically validated and reproducible.
- **Attribution:** Any novel formulas, signal processing algorithms, or third-party datasets must include formal academic citations.

---

## 🛠 Development Workflow

### 1. Environment Setup

We recommend using [`uv`](https://github.com/astral-sh/uv) or standard Python virtual environments:

```bash
# Clone the repository
git clone https://github.com/dyneresearch/DopplerGPS.git
cd DopplerGPS

# Create and activate virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate

# Install in editable mode with development dependencies
uv pip install -e ".[dev]" || uv pip install -e .
uv pip install pytest pytest-cov black ruff mypy
```

### 2. Running Tests

Every contribution must pass the test suite:

```bash
# Run complete test suite
pytest tests/ -v

# Run with execution timing
pytest tests/ -v --durations=10
```

### 3. Code Quality & Style

- **Python 3.10+ Compatibility:** Ensure code runs across Python 3.10, 3.11, and 3.12.
- **Type Annotations:** All public functions and classes must include complete type hints (`typing` / `from __future__ import annotations`).
- **Docstrings:** Follow Google or NumPy style docstrings for all mathematical formulations, explaining coordinate frames (e.g. WGS84, ECEF, ENU) and units ($m$, $s$, $\text{Hz}$, $^\circ\text{C}$).
- **Formatting:** Code should be formatted with `black` and checked with `ruff`.

---

## 🚀 Submitting Pull Requests

1. **Fork the repository** and create a feature branch (`git checkout -b feature/acoustic-refinement`).
2. **Add unit tests** under `tests/` covering new mathematical models, schema parsers, or edge cases.
3. **Verify tests pass locally** before pushing.
4. **Open a Pull Request** against `main` on `dyneresearch/DopplerGPS` with:
   - A clear explanation of the mathematical derivation or algorithmic change.
   - Benchmark timings or performance impact (if modifying DuckDB queries or STFT scoring).
   - Before/after spectrogram overlays or test outputs if modifying visualization logic.

---

## 🔒 Responsible Security & Privacy Disclosure

If you discover a vulnerability, potential privacy issue, or unintended exploitation vector in DopplerGPS, please do **not** open a public GitHub issue. 

Instead, submit a confidential report to our security team:
* **Email:** [connect@dyneresearch.com](mailto:connect@dyneresearch.com)
* **Subject:** `[SECURITY DISCLOSURE] DopplerGPS - <Summary>`

We will review, acknowledge receipt within 48 hours, and coordinate responsible remediation before public disclosure.

---

## 🏢 About Dyne Research

Dyne Research is an independent research lab advancing geospatial computing, applied physics, and open-source intelligence.  
Visit [dyneresearch.com](https://dyneresearch.com) or contact [connect@dyneresearch.com](mailto:connect@dyneresearch.com).

