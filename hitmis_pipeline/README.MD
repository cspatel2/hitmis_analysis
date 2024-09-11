# HiT&MIS Level 1A Data Generator
Generates per-day, mosaic-segregated, time-series of line-straightened data from RAW FIT files.

## Usage
```sh
$ python /path/to/hitmis_l1a_converter.py <rootdir> [destdir]
```

`rootdir` must be a directory containing FIT files in folders named `YYYYMMDD`, i.e. FIT files must be pre-sorted into per-day folders.
The HiT&MIS programs already do this.

To get help on the level 1a converter:
```sh
$ python /path/to/hitmis_l1a_converter.py --help
```