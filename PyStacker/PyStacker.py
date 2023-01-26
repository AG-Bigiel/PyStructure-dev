"""
This is a wrapper to perform stacking either using an existing PyStructure, or prepared input 3D cubes.

MODIFICATION HISTORY
    -   v1.0 21 January 2023
"""
__author__ = "J. den Brok"
__version__ = "v1.0"
__email__ = "jakob.den_brok@cfa.harvard.edu"
__credits__ = ["L. Neumann","M. Jimenez-Donaire", "E. Rosolowsky","A. Leroy "]


#general modules needed for routines
import numpy as np
import pandas as pd
import os.path
from os import path
import shutil
from astropy.io import fits
from datetime import date
import argparse
today = date.today()
date_str = today.strftime("%Y_%m_%d")

#import relevant PyStructure and PyStacker functions
import sys
sys.path.append("./scripts_PyStructure/")
sys.path.append("./scripts_stacking/")
from stacking import *

from create_database import *


#Step 1: Prepare ------------------------------------------------
#require input of config file
parser = argparse.ArgumentParser(description="Run PyStacker")
parser.add_argument("-c","--config",required = True)
parser.add_argument("-m","--mode",required = True,choices=['PyStruc', '3D_cube'], type = str, help = "PyStruc or 3D_cube")
args, leftovers = parser.parse_known_args()


conf_file = args.config
mode = args.mode


#generate a folder for temporary files
if os.path.exists("./Temp_Files/"):
    shutil.rmtree('./Temp_Files')
os.makedirs("./Temp_Files/")


if mode == "PyStruc":
    #separate config into required and optional parametrs
    
    py_input ='./Temp_Files/conf_Py.py'
    py_input_opt ='./Temp_Files/conf_Py_opt.py'
    
    loc = 0
    with open(conf_file,'r') as conf_f, open(py_input,'a') as req_f, open(py_input_opt,'a') as opt_f:
        for line in conf_f:
            if "Step 2:" in line:
                loc = 1
            if loc == 0:
                req_f.write(line)
            elif loc ==1:
                opt_f.write(line)
            
    #add a line to see all variables defined in optional
    with open(py_input_opt, 'a') as file:
        file.write("opt_variables = dir()")
    #import and use variables from config_file
    sys.path.append("./Temp_Files/")
    from conf_Py import *
    from conf_Py_opt import *

    #iterate over optional parameters
    kwargs={}
    for name in opt_variables:
        # Print the item if it doesn't start with '__'
        if not name.startswith('__'):
            kwargs[name] = eval(name)
    #run the stacking code
    get_stack(fnames, prior, lines, final_direc, dir_data = data_direc, xtype = xtypes, **kwargs)



elif mode == "3D_cube":
    py_input ='./Temp_Files/conf_Py.py'
    py_input_opt ='./Temp_Files/conf_Py_opt.py'
    stack_file = './Temp_Files/stack_temp.txt'
    cube_file = './Temp_Files/cube_list_temp.txt'
    
    loc = 0
    with open(conf_file,'r') as conf_f, open(py_input,'a') as req_f, open(py_input_opt,'a') as opt_f, open(cube_file,'a') as cube_f,open(stack_file,'a') as stack_f:
        for line in conf_f:
            if "Step 4:" in line:
                loc = 1
            elif "Step 5:" in line:
                loc = 2
            elif "Step 6:" in line:
                loc = 3
                
            if loc == 0:
                req_f.write(line)
            elif loc ==1:
                stack_f.write(line)
            elif loc ==2:
                cube_f.write(line)
            elif loc ==3:
                opt_f.write(line)
            
    #add a line to see all variables defined in optional
    with open(py_input_opt, 'a') as file:
        file.write("opt_variables = dir()")
    #import and use variables from config_file
    sys.path.append("./Temp_Files/")
    from conf_Py import *
    from conf_Py_opt import *

    #iterate over optional parameters
    kwargs={}
    for name in opt_variables:
        # Print the item if it doesn't start with '__'
        if not name.startswith('__'):
            kwargs[name] = eval(name)
           
    input = [sources, ra_ctr, dec_ctr, posang_deg,incl_deg,dist_mpc,target_res, max_rad, data_direc,spacing_per_beam,velocity_map, "./Temp_Files/", [NAXIS_shuff, CDELT_SHUFF]]
    
    #create the file
    fnames = create_database(input,stack_file,cube_file,)
    head_tail = path.split(fnames[0])
    #add the shuffeling
    
    get_stack([head_tail[1]], prior, lines, final_direc, dir_data = head_tail[0]+"/", xtype = xtypes, **kwargs)
    #perform the stacking
    

#remove the temporary folder after the run is finished
shutil.rmtree('./Temp_Files')
    
    
