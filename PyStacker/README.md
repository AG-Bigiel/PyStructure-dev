# PyStacker

`PyStacker` is a wrapper that performs the spectral stacking code on either an existing `PyStructure` database or a set of 3D data cubes (fits files). When executing via the terminal the `-c` parameter sets the configure file and `-m` the mode.

## Instruction:

There exist two modes how the PyStacker code can be utilized:

1. **With a `PyStructure` Database**

> In case a database (as a `.npy` file) exists, the code can be executed by adjusting the parameters in the configur file as shown in `run_PyStructure.conf`. Then simply execute in the terminal with:
> ```
> python PyStacker.py -c run_PyStructure.conf -m PyStruc
> ```
> where `PyStruc` sets the mode on how to execute the program.

> **Note:** You can provide optional parameters (list: TBD). You can also simply remove these parameters (or add pre-existing optional parameters) at the end of the configure file.

2. **Only with 3D Fits Datacubes**
> The code now also works without a pre-existing PyStructure database. 3D Datacubes are required as input. These have to be convolved and regridded to the same resolution and the same grid. In addition, a velocity map with which to shuffle the spectra, must be provided. It can be executed by adjusting the parameters in the `run_3DCube.conf` file:
> ```
> python PyStacker.py -c run_3DCube.conf -m 3D_cube
> ```
> where `3D_cube` sets the mode on how to execute the program.

