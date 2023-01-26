# PyStacker

`PyStacker` is a wrapper that performs the spectral stacking code on either an existing `PyStructure` database or a set of 3D data cubes (fits files)

## Instruction:

There exist two modes how the PyStacker code can be utilized:

* **With a `PyStructure` Database**

In case a database (as a `.npy` file) exists, the code can be executed by adjusting the parameters in the configur file as shown in `run_PyStructure.conf`. Then simply execute with:
```
python PyStacker.py -c  run_PyStructure.conf -m PyStruc
```

where `PyStruc` sets the mode on how to execute the program.
