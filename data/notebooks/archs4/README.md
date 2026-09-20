# ARCHS4 Preparation

These scripts convert the upstream ARCHS4 HDF5 source into the harmonized
AnnData matrix used by the bulk pretraining builder. Review their input names
and resource requirements before execution; the source is too large for most
local machines.

- `archs4.py` performs the initial extraction and conversion.
- `archs4_2.py` contains the subsequent preparation stage.
- `run_archs4.sh` and `run_archs4_2.sh` are Slurm wrappers for these stages.

Set `SCBFM_ROOT_DIR` to the data workspace, `SCBFM_REPO_DIR` to the checkout,
and `SCBFM_SIF` to the core image.
Submit from the desired log directory. Check that the first stage completed
before running the second. Input/output filenames are specified near the top
of each Python script. Source acquisition is separate from these conversions.
