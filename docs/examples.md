# More examples

Two tutorials run on every build of this site, and every number on them is the output of the
code above it: {doc}`Validate NumPy arrays <examples/array_checks>` and
{doc}`Custom checks and read-only arrays <examples/custom_operations>`.

Two further files fit declarations to the shape of a real consumer. They ship with the package,
and this site does not execute them:

| File | What it is |
|---|---|
| [forecast_scale.py](https://github.com/egordm/tenspec/blob/v0.1.0/packages/tenspec/examples/forecast_scale.py) | declarations fitted to the method shape of a real forecasting scaler, on NumPy |
| [locos_capture.py](https://github.com/egordm/tenspec/blob/v0.1.0/packages/tenspec/examples/locos_capture.py) | declarations fitted to the operand relationships of a real attribution run, on PyTorch |

`locos_capture.py` composes storage with older Pydantic validators. It is not the native
storage path: for that, see
[An owned computed result](pydantic-models.md#an-owned-computed-result).

Run one from a clone of the repository:

```bash
uv run python packages/tenspec/examples/forecast_scale.py
```
