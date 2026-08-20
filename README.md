# Bifurcation Handling for Reduced-Order Modeling of CCO Vascular Trees

Developed by Natalia Rubio as part of Ph.D. at Stanford University.  Subject to Stanford IP policy.  Please do not delete or move this repository without notifying me (Natalia).

Workflow for using the [RRI junction handling method](https://www.sciencedirect.com/science/article/abs/pii/S0010482524015051) on [CCO-style trees](https://www.science.org/doi/10.1126/science.adj6152), as described in the paper, [Data-driven bifurcation handling in physics-based reduced-order vascular hemodynamic models](https://www.sciencedirect.com/science/article/abs/pii/S0169260725006455).  


The junction elements of 0D "electric circuit" models of vascular trees are modified to include linear and quadratic resistors as well as inductors.  The resistances and inductances of the junctions are predicted by neural networks from a vector describing the junction geometry.  The neural networks are trained on synthetic 3D bifurcation flow data generated in an automated fashion using an the [SimVascular Python API](simvascular.github.io/documentation/python_interface.html) for geometry and mesh generation and svFSI now [svMultiPhysics](https://github.com/SimVascular/svMultiPhysics) on the Stanford HPC cluster [Sherlock](www.sherlock.stanford.edu/docs/user-guide/ondemand/).


## Repository layout

| Path | Role |
|------|------|
| `trees/` | CCO tree geometry and simulation data: `geo_files/`, `zerod_input/`, `zerod_output/`, `zerod_output_cent/` (0D results on centerlines), `threed_results/`, `threed_output_cent/`, `standard0d_input_file_generators/`, etc. |
| `data/` | Training data for neural networks : `jax_arrays/`, `scaling_dictionaries/`, `split_indices/`, `tensors/`, `synthetic_junctions/`, etc. |
| `results/` | Trained networks, visualizations, and experimental results. |
| `util/` | All Python functionality (see below). |

## `util/` modules (overview)

- **`util/zerod/`** — Build and transform **0D solver** inputs (`solver_0d.json`): standard networks and variants. Optimization with **CasADi** replaces svZeroDSolver for the forward 0D solution for more robust convergence.
- **`util/svFSI/`** — 3D flow simulations for training data generation. Writing solver files, launching steady/unsteady jobs (often via SLURM/`sbatch`), projecting solutions onto centerlines, averaging fields, resolution sweeps, and exporting reduced solution sets at junction offsets.
- **`util/tree/`** — Extract resistances, junction dictionaries, and branch structure from 0D/centerline data; helpers for **energy checks** and tree-level analysis.
- **`util/centerline_projection/`** — Task-style pipeline (multi-fidelity estimation, calibration, 3D simulation hooks) and **`project_0d_to_3d.py`**: projects **0D** branch results onto **3D** centerlines (`centerlines.vtp`) for visualization and comparison.
- **`util/fidelity_comparison/`** — Compare **0D vs 3D** (nodal, inlet, centerline).
- **`util/synthetic_data_processing/`** — End-to-end **dataset build**: extract features from simulations, train/val splits, scaling, export **JAX** arrays (`process_synthetic_data.py` orchestrates several steps).
- **`util/neural_net/`** — **JAX / Optax** MLP training for junction coefficient targets (e.g. **rri** / **ri** / **rr** output modes in `nn_model.py`).
- **`util/tools/`** — Shared utilities (**VTK** I/O, junction geometry, BC helpers) used across the above.
- **`util/bif_gen/`**, **`util/bifurcation/`**, **`util/analysis/`** — Bifurcation generation, geometry, and analysis scripts.
- **`util/CCO_sherlock/`** — Cluster-oriented helpers and duplicated projection utilities for remote runs.

## Typical workflows (conceptual)

1. **Generate training data** — Build supervised datasets of bifurcation geometries and flow simulations, then process for ML training.  
Relevant files: `util/synthetic_data_processing/process_synthetic_data.py`, `util/synthetic_data_processing/synthesize_synthetic_data.py`, `util/synthetic_data_processing/extract_synthetic_data.py`, `util/synthetic_data_processing/extract_synthetic_data_unsteady.py`, `util/synthetic_data_processing/train_val_split.py`, `util/synthetic_data_processing/get_jax_arrays.py`, `util/synthetic_data_processing/get_scaling_dict.py`

2. **Train neural networks** — Train/evaluate JAX models that predict junction coefficients (RRI/RI/RR variants) from geometry features.  
Relevant files: `util/neural_net/launch_training.py`, `util/neural_net/train_nn.py`, `util/neural_net/nn_model.py`, `util/neural_net/nn_util.py`, `util/neural_net/examine_model.py`, `util/neural_net/launch_training_ray_opt.py`

3. **Generate and run 3D simulations on CCO trees** — Create svFSI solver setups, launch steady/unsteady runs, and post-process centerline-projected reduced solutions.  
Relevant files: `util/svFSI/write_solver_files.py`, `util/svFSI/write_solver_files_unsteady.py`, `util/svFSI/launch_steady_jobs.py`, `util/svFSI/launch_unsteady_jobs.py`, `util/svFSI/launch_tree_sims.py`, `util/svFSI/get_avg_sol.py`, `util/svFSI/get_avg_sol_unsteady.py`, `util/svFSI/export_various_offsets.py`, `util/svFSI/export_various_offsets_unsteady.py`, `util/svFSI/centerline_proj.py`

4. **Run 0D simulations on CCO trees (standard vs RRI)** — Generate baseline `solver_0d.json`, convert to enhanced junction models, then run and compare standard vs modified 0D predictions.  
Relevant files: `util/zerod/get_inp_standard.py`, `util/zerod/correct_BCs.py`, `util/zerod/standard_to_RRI_full_fit.py`, `util/zerod/standard_to_RRI_junctions_fit.py`, `util/zerod/standard_to_RRI_nn_branch_fit.py`, `util/zerod/svzerod_to_casadi.py`, `util/zerod/test_junction_model.py`, `util/zerod/test_junction_model_single.py`
