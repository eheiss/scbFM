# Tests

From the repository root, in the Python 3.11 development environment:

```bash
python -m pytest -q
```

Tests use synthetic matrices, small models, and mocked external components.
They cover fold reuse and alignment, expression tokenization, training
correctness, metric calculations, deconvolution post-processing, analysis
archives, configurable paths, and argument forwarding through the cluster
launcher. They do not require datasets, external checkpoints, Slurm, or a GPU.

Checks against personal submission scripts outside the checkout have been
replaced by tests of the public launcher and runner configuration. Full
multi-GPU training, container builds, and reproduction of the reported thesis
scores still require their respective cluster environments and input artifacts.
