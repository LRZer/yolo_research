# Oracle-PGB real VOC visualizations

These are real images from the fixed VOC development split, processed with the
same registered Oracle geometry and renderer used in the P1 Oracle experiment:
10% box margin, 0.5 box-scale transition, 8-pixel minimum radius, Gaussian
levels 0/1/2/4/8 and residual mix 0.5.

- `original_640/`: clean letterboxed detector inputs.
- `strength_map/`: blue protects detail (strength near 0); red applies stronger
  blur (strength near 1).
- `oracle_processed/`: clean processed detector inputs, without drawn boxes.
- `absolute_difference/`: heatmap of the pixel change.
- `comparison_panels/`: 2x2 explanation panels with ground-truth boxes.
- `manifest.csv`: source paths, classes and quantitative processing statistics.

One distinct image was selected for each of the 20 VOC classes, preferring
multi-object examples. Ground-truth boxes are used only because this is an
Oracle diagnostic; this cannot be reproduced at ordinary inference time.
