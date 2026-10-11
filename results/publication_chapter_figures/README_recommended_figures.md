# Recommended Chapter Figure Set

Use these figures in the chapter:

1. `fig01_topology_geometry_fidelity`: main quantitative topology/geometry proof.
2. `fig02_lesion_capture_ablation`: atlas/style rendering diagnostic. The final bar is Full CCLD + Atlas/Style, not raw Full CCLD.
3. `fig03_structural_validity_modes`: compact failure-mode plot for empty/tiny/valid masks.
4. `fig04_mask_morphology_diagnostics`: replaces the plain black-white mask grid with contour, bounding box, centroid, medial-axis proxy, area, Betti-0, and compactness labels. If generated from the atlas v3 folder, the final row must be read as atlas/style-refined.
5. `fig05_rgb_mask_alignment_plate`: qualitative image-mask synthesis comparison, with atlas/style-refined Full CCLD labeled separately when applicable.
6. `fig06_downstream_segmentation_utility`: optional but useful for the utility section.

Avoid using raw black-background binary-mask grids as main figures. Keep them only as supplementary diagnostics if needed.
