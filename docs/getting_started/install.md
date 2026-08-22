# Install LMX

LMX supports Python 3.10–3.13.

```console
python -m venv .venv
source .venv/bin/activate
python -m pip install lmx
lmx --help
```

Install plotting support with `python -m pip install "lmx[visualization]"`.
The standard JAX package uses the CPU. For NVIDIA, AMD, or Apple accelerator
options, follow the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
and verify the selected devices:

```console
python -c "import jax; print(jax.devices())"
```

For source development:

```console
git clone https://github.com/uwplasma/LMX.git
cd LMX
python -m pip install -e ".[dev,docs]"
python scripts/run_full_test_suite.py
```

LMX imports lazily, so `import lmx` does not initialize JAX. A persistent
compilation cache is useful for repeated cases:

```python
import lmx
lmx.enable_compilation_cache(".jax-cache")
```
