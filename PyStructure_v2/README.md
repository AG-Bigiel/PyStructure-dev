# The PyStructure2.0 Code
**Initialize and fill in the database as a python dictonary**

This repository contains the python scripts that sets up and generates a structure (or rather a python dictionary). It constitutes a major upgrade with respect to Version 1. 

## How to run it

* **Option a)**: Run it in the same way as Version 1 (see Version 1 for details).

* **Option b)**: Use config file to set everything up

## The config file

If you work with the config file, you do not need to make any changes to the `create_database.py` file.

* **Step 1**: Make sure your galaxy is listed in the `geometry.txt` file (No need to comment out galaxies that are not used).

* **Step 2**: Follow the description of the steps in the `PyStructure.conf` file and define the variables accordingly.

* **Step 3**: Run in the terminal with the following command:

`python3 create_database.py --config PyStructure.conf`
