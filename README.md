# IPHASMosaic
Software that create mosaics using the IPHAS images with python and montage package.

### Pre-requisites
- Python 3.7
- Unix based systems (on Windows, it was tested under WSL)

### Installation
Download the latest wheel located in the `dist` directroy and run `pip install name_of_the_wheel.whl`. 

### Usage
```python
from astropy.coordinates import SkyCoord
from iphasmosaic import IPHASMosaic

coors1 = SkyCoord("19:05:48", "+10:30:00", unit=("hourangle", "deg"))
im = IPHASMosaic(name="StDr19", band="halpha", ra=coors1.ra.value, dec=coors1.dec.value, radius=120)
im.create_tree()
im.download_and_prepare(qcgrade=["A++", "A+", "A"], remove_duplicates=True, report_table=None)
im.mosaic(fix_nan=True)
```

This will generate a final image in the desired filter and save it as `.fits` under the main directory (e.g., `./StDr19`).