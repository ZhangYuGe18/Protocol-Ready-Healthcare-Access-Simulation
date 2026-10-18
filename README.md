# Protocol-ready access model

This repository contains the reusable computational model for studying how uncertain travel, preparation behavior, and clinic resource queues interact with protocol readiness. It implements an agent-based, discrete-event simulation (ABM–DES), an adaptive preparation policy, calibration utilities, and feasibility-frontier calculations.

No patient records, observed outcomes, route coordinates, API credentials, data-loading scripts, or study-specific configuration and result files are included. The example and tests use fictional values only; their outputs are not research findings and the example parameters are not calibrated estimates.

## Contents

- `protocol_access/` — simulation, event scheduling, state model, adaptive policy, calibration mathematics, and frontier methods.
- `examples/synthetic_demo.py` — a small, entirely synthetic run.
- `tests/` — model invariants and reproducibility checks using synthetic fixtures.

## Run

Use Python 3.10 or newer. From this directory:

```bash
python -m pip install -r requirements.txt
python -m examples.synthetic_demo
python -m unittest discover -s tests
```

The public API is `protocol_access.simulate_cohort(frame, pars, config, scenario, seed)`; `frame` is an in-memory `pandas.DataFrame`. See `examples/synthetic_demo.py` for the expected abstract columns and configuration keys. No file path, data connector, or external service is required. The model retains simulated queue dynamics, but no observed queue logs or route-configuration counts are shipped.

This software is intended for methodological exploration and aggregate scenario comparison, not individual clinical decisions. To reproduce a particular study, authorized investigators must separately supply and validate their own data and calibrated parameters.

中文说明：[README_zh-CN.md](README_zh-CN.md)
