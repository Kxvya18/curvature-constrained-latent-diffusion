# Final Chapter Plot Captions

Figure 1. Topology and geometry fidelity across 256 generated masks. Full CCLD eliminates Betti-0, Euler, and H0 persistence errors while retaining the lowest Forman-Ricci curvature discrepancy among topology-preserving variants.

Figure 2. Structural validity failure modes. Baseline latent diffusion frequently produces empty or tiny masks, whereas CCLD variants maintain valid lesion-scale area and avoid border-contact failures.

Figure 3. Lesion-scale structure capture in the atlas/style rendering setting. The final row should be named Full CCLD + Atlas/Style because atlas refinement replaces the raw generated mask before rendering.

Figure 4. Image realism must be interpreted jointly with structure. Baseline obtains favorable FID partly by making minimal lesion edits; Full CCLD and the atlas/style-refined rendering should be reported separately.

Figure 5. Downstream segmentation utility. Augmenting with Full CCLD samples improves held-out IoU and boundary F1 compared with baseline-generated augmentation.
