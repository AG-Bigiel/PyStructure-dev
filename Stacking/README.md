# Stacking Code

The stacking code allows a user to stack spectra of a cube with respect to a predefined quantity, such as the galactocentric radius, star formation rate density and more. Here we give a short description and step-by-step guide on how to use the code for an optimal experience. 

* The main file is the `stacking.py` file. 
* The `how_to_stack.ipynb` is an Jupyter notebook with an example of how to run the code
* The folder `Example` contains an `IDL` structure with the data to stack as well as the results of the example run

## General Structure of the Code
The `stacking.py` file contains the main part of the code which call upon all the other side files. The procdure is the following:

1. It is specified by which quantity will be stacked (by the `xtype` variable).
2. The to be stacked quantity is binned acording to the values given in `bin_values.txt` using the subscript `stacking_func.py`.
3. The spectra are stacked for each bin using the functions in `stack_specs.py`.
4. We fit a Gaussian to the stacked spectra to get better constraints of the intesnity of the line. For this the fitting procedure in `fitting_routine.py` is used.

## Tutorial
In order to provide an optimal experience with the code, have a look at the `how_to_stack.ipynb`  notebook. In there is a step--by--step guide on what to do to get your spectra stacked.


