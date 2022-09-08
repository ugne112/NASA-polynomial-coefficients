# About
This is an MSc project to calculate updated 7-coefficient NASA polynomials for heat capacity data from ExoMol molecular line lists.

# Getting started

## Prerequisites
This program requires Anaconda or Miniconda to make use of the Numba package. Once either is installed, install Numba
```
conda install numba 
```
Python packages used in these codes are:
sys
os
numpy
pandas
json
matplotlib

Please ensure these are installed before running the modules.

## Download
Download the contents locally
```
git clone https://github.com/ugne112/NASA-polynomial-coefficients.git
```

# Usage

Once everything is set up locally, polynomial coefficients can be calculated!

Navigate inside the directory to run

```
python polynomial_coefficients.py
```
Which is when you will be asked to enter a species name (currently, sample data exists for H2O and CO only) and temperature range

<img width="682" alt="Screenshot 2022-09-08 at 00 55 51" src="https://user-images.githubusercontent.com/71969506/188990793-e98774b3-4771-427d-96cf-d6ad534a1712.png">

Upon entering which, your calculation will be underway.

Upon completion, you will receive a message:

<img width="528" alt="Screenshot 2022-09-08 at 01 04 39" src="https://user-images.githubusercontent.com/71969506/188991979-fa7e9d82-b5af-491f-be6a-51b3f969e52c.png">

Onece you have run the polynomial_coefficients.py calculation for both temperature ranges for a given species, you can calculate heat capacity data from coefficients and produce a heat capacity comparison plot by running:
```
python Cp_plot.py
```
You will be asked to enter species name (again, sample data for H2O and CO only).

Upon completion, the plot will come up.

