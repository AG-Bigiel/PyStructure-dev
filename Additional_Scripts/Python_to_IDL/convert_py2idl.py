import pidly
import numpy as np
import os
import argparse
import sys

def update_progress(job_title, progress):
    """
        From: https://blender.stackexchange.com/questions/3219/how-to-show-to-the-user-a-progression-in-a-script
        Function that shows progress
        :param job_title: Str displayed in the Console
        :param progress: Fraction between 0 and 1
    """
    length = 20 # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
    if progress >= 1: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()



parser = argparse.ArgumentParser()
parser.add_argument("path",help="npy file path")
args = parser.parse_args()

#idl = pidly.IDL()
idl = pidly.IDL(long_delay=0.05)


path = "ngc5194_data_struct_23as_2020_11_11.npy"
struct = np.load(str(args.path),allow_pickle = True).item()


idl("struct = {start: !values.f_nan}")

print("[INFO]\t Start Conversion from Python to IDL ...")

for i,key in enumerate(struct.keys()):
    update_progress("\t Converting",i/len(struct.keys()))
    # if parameter is just a float, int, str, use the pidly function to add keyword
    if not isinstance(struct[key], np.ndarray) and not isinstance(struct[key], list):
        #idl key cannot include "-", "+", etc.
        idl_key = key.replace("-","_") 
        data = {idl_key: struct[key]}
        tag_new = idl._python_to_idl_structure(data, "struct_add")
        idl(tag_new)
        idl("tags = tag_names(struct)")
        idl('tot = total(tags eq '+'"'+key.replace("-","_")+'"'+')')
        # check if tag is not already defined
        if idl.tot==0:
            idl("struct = create_struct(struct, struct_add)")
        else:
            idl('struct.(where(tags eq "'+idl_key+'")) = '+tag_new.split(":")[-1].split("}")[0])
    
    #if paramezer is a list
    elif isinstance(struct[key], list):
        idl_key = key.replace("-","_")
        with open('listfile.txt', 'w') as filehandle:
            for listitem in struct[key]:
                filehandle.write('%s\n' % listitem)
    
        if isinstance(struct[key][0], str):
            idl('readcol,"'+"listfile.txt"+'" , list_file, format="A", /SILENT ')
        else:
             idl('readcol,"'+"listfile.txt"+'" , list_file, /SILENT ')
        idl("struct_add = {"+idl_key+": list_file}")
        
        idl("tags = tag_names(struct)")
        idl('tot = total(tags eq '+'"'+key.replace("-","_")+'"'+')')
        # check if tag is not already defined
        if idl.tot==0:
            idl("struct = create_struct(struct, struct_add)")
        else:
            idl('struct.(where(tags eq "'+idl_key+'")) =list_file ')
        os.remove("listfile.txt")

    elif isinstance(struct[key], np.ndarray):
        sz_key = np.shape(struct[key])
        shape_key = len(sz_key)
        idl_key = key.replace("-","_")

        if shape_key==1:
            np.savetxt("temp_file.txt", struct[key])
            
            
            idl('readcol,"'+"temp_file.txt"+'" , temp_file, /SILENT ')

            idl("struct_add = {"+idl_key+": temp_file}")
            idl("tags = tag_names(struct)")
            idl('tot = total(tags eq '+'"'+idl_key+'"'+')')
            # check if tag is not already defined
            
            if idl.tot==0:
                idl("struct = create_struct(struct, struct_add)")
            else:
                idl('struct.(where(tags eq "'+idl_key+'")) =temp_file ')

            os.remove("temp_file.txt")
        if shape_key==2:
            npts = sz_key[0]
            n_chn = sz_key[1]
            idl("empty_spec = fltarr("+str(npts)+","+str(n_chn)+")* !values.f_nan")
            for n in range(npts):
                np.savetxt("temp_file.txt", struct[key][n,:])
                idl('readcol,"'+"temp_file.txt"+'" , this_spec, /SILENT ')
                idl("empty_spec["+str(n)+",*] = this_spec")
                os.remove("temp_file.txt")
                

            idl("struct_add = {"+idl_key+": empty_spec}")
            idl("struct = create_struct(struct, struct_add)")
    else:
        print("[ERROR]\t"+key+" Type not clear")

idl('save, file="'+path.replace(".npy", ".idl")+'" , struct')
update_progress("\t Converting",1)
    




