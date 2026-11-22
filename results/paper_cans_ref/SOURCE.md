Cans-scene panoramas from the paper (comparison only, NOT ground truth): Park, Holynski, Seitz,
"Seeing the World in a Bag of Chips", CVPR 2020, arXiv 2001.04642v2, supplementary Figure 13, page 13.
Extracted at native resolution with `pdfimages -f 13 -l 13 -j` from ../p2r_paper/source_paper_arxiv2001.04642v2.pdf; not modified.

- fig13a_cans_input_texture.jpg  (a) Input Texture: the same two cans (Pepsi, Diet Coke) on the carpet as cans_data/
- fig13d_cans_srm_M2.jpg         (d) Recovered SRM, M = 2 material bases, 500 x 175 (vertically cropped in the figure)
- fig13e_cans_srm_M3.jpg         (e) Recovered SRM, M = 3 material bases, 500 x 175

The paper shows no ground-truth environment for Cans (ground truths exist only for the scenes in Fig. 5 and 6).
Azimuth origin and handedness relative to our lookup layout are unknown: compare with an azimuth search (envmap.compare(..., align=True)) and try a horizontal flip.
