# figures/

Output directory for paper figures produced by notebooks in this repo.

## Convention

Notebooks save figures here via `fishsense_imwut.figures.save_paper_figure`:

```python
from fishsense_imwut.figures import save_paper_figure

save_paper_figure(fig, "laser_spot_path.pdf")
```

Filenames should match the basename in the paper's
`\includegraphics{Images/graphs/<name>}` reference, so artifacts can be
copied straight into the paper source's `Images/graphs/` directory.

## Paper figures expected here

| File                       | Paper reference          | Producing notebook                                                  |
| -------------------------- | ------------------------ | ------------------------------------------------------------------- |
| `laser_spot_path.pdf`      | `fig:laser_spot_path`    | `notebooks/field_reconstruction/laser_spot_path.ipynb`              |
| `refraction_error.pdf`     | `fig:refraction_error`   | `notebooks/optics_simulation/flat_port_refraction.ipynb`            |
| `field_fish_lengths.png`   | `fig:field_fish_lengths` | _TBD_ — §4 Results, real-field fish-length distribution             |
