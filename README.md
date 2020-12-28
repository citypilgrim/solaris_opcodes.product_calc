# solaris_opcodes.product_calc

Product computation code for Scanning Micro-pulser LIDAR

## Usage

### Computing NRB products and returns a dictionary of data

Adjust the data directory and timings accordingly and run

```
python -m solaris_opcodes.product_calc.nrb_calc
```

### Computing all lidar products and returns a dictionary of data

Adjust the data directory and timings accordingly and run

```
python -m solaris_opcodes.product_calc
```

### Searching for optimal scan time

To search backwards for the latest timing in which a specific lidar data would give you a complete sweep of it's scan pattern, specify the correct details in the script and run

```
python -m solaris_opcodes.product_calc.optimaltime_search
```

### Analysing individual lidar profiles in cloud product generation

Only the Gradient Based Cloud Detection Method (GCDM) is implemented
Adjust the data directory and timings accordingly and run

```
python -m solaris_opcodes.product_calc.cloud_calc.gcdm.gcdm_{original,extended}
```

---

### Computing rayleigh profile from `.cdf` file

Default file is valid for Singapore.

```
python -m solaris_opcodes.product_calc.constant_profiles.rayleigh_gen
```

### Plotting calibration profiles

To plot both overlap and afterpulse calibration profiles, ensure that the calibration profile data are in the specified data directory and run. This plots out the computed calibration profiles against the profiles computed by the SigmaMPL software which are stored in `.csv` files

```
python -m solaris_opcodes.product_calc.cali_profiles
```

To plot the afterpulse profiles generated from `.mpl`/`.csv` files with their corresponding uncertainties, using various uncertain propagation methods, first ensure that the calibration profile data is in the specified data directory, and then run

```
python -m solaris_opcodes.product_calc.cali_profiles.afterpulse_{csv,mpl}gen
```

To plot the overlap profiles generated from `.mpl`/`.csv` files with their corresponding uncertainties, using various uncertain propagation methods, first ensure that the calibration profile data is in the specified data directory, and then run

```
python -m solaris_opcodes.product_calc.cali_profiles.overlap_{csv,mpl}gen
```

To compute the detector deadtime coeffcients from the experimental calibration values and save them into an appropriate directory, first make sure that the details in the script are correct, then run

```
python -m solaris_opcodes.product_calc.cali_profiles.deadtime_gen
```