# LHS-Compatible-model

This repository contains code related to numerical integration using Lebedev points of order 131 and analysis of compatible models for noisy Positive Operator-Valued Measures (POVMs). It includes Python scripts that demonstrate various aspects of spherical numerical integration and POVM compatibility.

## Files

- `Lebedev_points_131.txt` and `Lebedev_weights_131.txt`: These files contain Lebedev points and corresponding weights used for spherical numerical integration of order 131.

- `compatiblemodel_NC.py`: This script provides an explicit construction of a compatible model for any 4-outcome noisy POVM. Users can define their own POVMs and compute an 18-effect POVM along with the corresponding response function.

- `linearsolver_guroby.py`: This example script illustrates that 14-effect POVMs may not always be sufficient for simulating the corresponding children POVM. The demonstration is carried out using linear programming, which is discussed as Farka's lemma in the methods section of our work.

