# deeptools_patched
A copy of the deepTools 3.3.2 Python package as packaged by Debian (python3-deeptools
3.3.2+ds-2; MIT, see LICENSE.txt) with one modified file,
`deeptools/plotHeatmap.py`: the middle region group (Shared) is drawn 1.3x taller, the gap
between the profile and the heatmap is widened (spacer heatmapwidth/8 -> /4), and the
region-group legend is placed outside the plot on the right. Every other module is identical to that package;
the package's test suite is omitted. `plotheatmap_patched.py` runs the patched plotHeatmap and
is called by `../run_peakheatmap.sh` and `../run_peakheatmap_matrix.sh` with
`PYTHONPATH` set to this directory.
