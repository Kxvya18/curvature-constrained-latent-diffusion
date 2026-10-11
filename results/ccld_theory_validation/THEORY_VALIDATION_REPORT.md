# CCLD Theory Validation Report

These are controlled mathematical stress tests for the CCLD operators.
They are not substitutes for real Kvasir synthesis experiments.

## Claim Status

- Betti projection theorem: directly tested by projecting fragmented and holed masks to target Betti anchors.
- Area projection bound: tested on random probability fields with target lesion-area ranges.
- Curvature fragmentation proposition: tested by comparing coherent ellipses with area-matched fragmented masks.
- First-order energy descent: tested on a differentiable structural surrogate using finite-difference gradients.

## Summary

{
  "area_topology_projection": {
    "n": 360,
    "success_rate": 0.9972222222222222,
    "area_error_mean": 0.009308026296110774,
    "area_error_std": 0.006047646492644535,
    "topology_success_rate": 1.0,
    "area_success_rate": 0.9972222222222222
  },
  "betti_projection": {
    "n": 320,
    "success_rate": 1.0
  },
  "betti_projection_with_hole": {
    "n": 320,
    "success_rate": 1.0
  },
  "curvature_fragmentation": {
    "n": 260,
    "success_rate": 1.0,
    "delta_fragment_minus_coherent_mean": 0.8959628199322618,
    "delta_fragment_minus_coherent_std": 0.15645678841251526,
    "coherent_curvature_l1_mean": 0.03860437409791221,
    "coherent_curvature_l1_std": 0.02038622952080613,
    "fragmented_curvature_l1_mean": 0.934567194030174,
    "fragmented_curvature_l1_std": 0.15609982720460983
  },
  "first_order_energy_descent": {
    "n": 120,
    "success_rate": 1.0,
    "energy_delta_mean": -0.0076743817016906805,
    "energy_delta_std": 0.0016495554694642247
  }
}

## Important Interpretation

Exact zero Betti errors are expected when hard projection succeeds. This validates the projection operator under its feasibility assumptions; it does not mean an unconstrained latent diffusion denoiser learned exact topology.

Area projection is an iterative discrete operator, so failures or residual errors identify where the theorem's termination/tolerance assumptions are not met.
