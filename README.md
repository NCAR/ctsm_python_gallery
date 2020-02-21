CTSM Python Gallery
===================

This repository includes a series of Jupyter Notebooks that highlight how Python can be used to analyze CTSM and CMIP-like model output.

## Notebooks

| Notebook      | Author        | Description  |
| ------------- |---------------| -------------|
| AnomalyDetrend.ipynb | Will Wieder | calculates gridcell anomalies from climatology and then detrends data |
| compare_esm_ssp_opt.ipynb | Katie Dagon | TODO |
| LandUseChange_maps.ipynb | Dave Lawrennce | TODO |
| MonthlySVD.ipynb | Will Wieder | complicated analysis looking at gridcell level singular value decomposition of GPP |
| ScatterExample.ipynb | Daniel Kennedy | TODO |
| SimpleExample.ipynb | Will Wieder | simple example to read in history file, calculate annual values & plot |
| WeightedMeans.ipynb | Katie Dagon | TODO |

# ctsm_py

Some of the notebooks in this gallery make use of some simple utilities stored in the `ctsm_py` module. Details on how to install and import `ctsm_py` are below:

## Installing

First clone this repository:

```
git clone https://github.com/NCAR/ctsm_python_gallery.git
```

Then intall the utilities:

```bash
cd ctsm_py
conda activate $ENVIRONMENT_NAME
pip install -e .
```

Import ctsm_py in Python

```python
In [1]: from ctsm_py.utils import weighted_annual_mean
```
