# uea_ucr_datasets

This package contains convenience functions and classes to access the UEA UCR
time series classification archive.

Currently it contains the following functionalities:
 - `Dataset` class: Loads UEA UCR dataset stored in the `sktime` format 
   from `~/.data/UEA_UCR/` or path provided via the `UEA_UCR_DATA_DIR`
   environment variable. This class is compatible with the pytorch `DataLoader`
   class.
 - `list_datasets`: List datasets available in the `~/.data/UEA_UCR/` folder or
   path provided via the `UEA_UCR_DATA_DIR`

## Example usage

Download the `sktime` version of the UEA and UCR datasets. And unpack them.
Move the folders of the individual datasets to the path `~/.data/UEA_UCR`.

```python
>>> import uea_ucr_datasets
>>> uea_ucr_datasets.list_datasets()
['LSST',..]
>>> d = uea_ucr_datasets.Dataset('UWaveGestureLibrary', train=True)
>>> first_instance = d[0]
>>> instance_x, instance_y = first_instance
```

## Alternative data paths

You can also store the data at another location, then it is required to set the
environment variable `UEA_UCR_DATA_DIR` appropriately such that the package can
find the datasets.
