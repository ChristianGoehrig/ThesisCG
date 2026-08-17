# ThesisCG — Snow Distribution & Terrain Analysis

Analysis code from a master's thesis studying the relationship between terrain
characteristics (slope, aspect, curvature, TPI, elevation) and snow depth
distribution, including snow-depth prediction from terrain features and
weather-station representativeness analysis.

This is research code written to support one thesis, not a packaged tool —
scripts are config-driven but expect the specific raster/vector datasets used
in that study. It is shared here for transparency and reference rather than
as a ready-to-run application. Expect ongoing changes as the underlying
research evolves.

## Layout

All code lives flat in the repo root:

| File | Role |
|---|---|
| `powdersearch.py` | Shared function library (raster I/O, terrain-parameter calculation, statistics, plotting) imported as `ps` by every other script. |
| `config_loader.py` | `ConfigLoader` class used by the snow-modelling script to load and validate `config_snow_modelling.yaml`. |
| `preprocessing.py` | Config-driven entry point: reprojects/aligns raw snow-depth rasters, computes timeseries statistics, difference maps, normalization, and terrain features. Driven by `config_preprocess.yaml`. |
| `runfile_representiveness.py` | Run-script for weather-station representativity analysis. Driven by parameters at the top of the file (and conceptually by `config_representativeness.yaml`). See note below — it depends on a module not included in this repo. |
| `SDD_spatial_terrain_paramert_elevationbased.py` | Spatial, elevation-stratified analysis of snow depth vs. terrain parameters. |
| `SDD_timeseries_per_terrain_parameter_variable.py` | Per-terrain-parameter timeseries analysis and classification (aspect/slope/curvature/TPI/geomorphons). |
| `snow_modelling_absolute_anaylsisplotupdate_update.py` | Main terrain-feature-to-snow-depth correlation and prediction script; trains a linear model on one avalanche outline/year and predicts on another. Driven by `config_snow_modelling.yaml`. |

## Configuration

Three YAML files drive the corresponding scripts:

- `config_snow_modelling.yaml` → `snow_modelling_absolute_anaylsisplotupdate_update.py`
- `config_preprocess.yaml` → `preprocessing.py`
- `config_representativeness.yaml` → `runfile_representiveness.py` (conceptually)

Every `paths:` entry in these files is left blank as a template — fill in
your own local data locations before running. `config_loader.py` and the
`load_config`/`validate_paths` helpers in `preprocessing.py` will raise a
clear error if a required path is missing or doesn't exist.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3 with `rasterio`, `rioxarray`, `xarray`, `geopandas`,
`shapely`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`,
`seaborn`, `cmcrameri`, and `pyyaml` (see `requirements.txt`). `geopandas`
in particular can be easier to install via conda than pip.

Each script appends `library_dir` (its own copy of the path to this repo) to
`sys.path` before `import powdersearch as ps` — set that variable (or the
`paths.library_dir` config entry) to wherever you've cloned this repo.

## Known limitation

`runfile_representiveness.py` imports `run_site_representativity_analysis`
from a `snow_site_analysis` module that is not part of this repository — it
predates a later refactor into `powdersearch.py` and was never fully ported.
The closest current equivalent is
`powdersearch.analyze_stations_with_circular_filter`, though its parameters
don't map one-to-one onto this run-script. Treat `runfile_representiveness.py`
as a reference for the intended parameters rather than a script that runs
out of the box.

## License

MIT — see [LICENSE](LICENSE).
