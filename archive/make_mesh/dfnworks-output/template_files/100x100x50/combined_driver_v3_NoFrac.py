##"""
#   :synopsis: Driver run file for TPL example
#   :version: 2.0
#   :maintainer: Jeffrey Hyman
#.. moduleauthor:: Jeffrey Hyman <jhyman@lanl.gov>
#"""

from pydfnworks import *
import os
import numpy as np
import subprocess
import pandas as pd 
import yaml

# LOAD VARIABLES FROM YAML
# ---------------------------------------
# Load the YAML file
with open('config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# Extract variables from the loaded config
variables = config['variables']

# Now you can use these variables in your code
# octree refinement
l = variables['l']
orl = variables['orl']
#general
flow_solver_in = variables['flow_solver_in']
ncpu_in = variables['ncpu_in']
DFN.params['seed']['value']=seed_value # create the same dfn each time

s_z = variables['s_z']
aq_z = variables['aq_z']
cb_z = variables['cb_z']
tot_z = variables['tot_z']
domain_size_x = variables['domain_size_x']
domain_size_y = variables['domain_size_y']
domain_size_z = variables['domain_size_z']
mat_perm_top= variables['mat_perm_top']
mat_perm_middle= variables['mat_perm_middle']
mat_perm_bottom= variables['mat_perm_bottom']
mat_por_top= variables['mat_por_top']
mat_por_middle= variables['mat_por_middle']
mat_por_bottom= variables['mat_por_bottom']
midlayer_size_x = variables['midlayer_size_x']
midlayer_size_y = variables['midlayer_size_y']
midlayer_size_z = variables['midlayer_size_z']
h = variables['h']
midlayer_center_x = variables['midlayer_center_x']
midlayer_center_y = variables['midlayer_center_y']
midlayer_center_z = variables['midlayer_center_z']
interface_bot = variables['interface_bot']
interface_top = variables['interface_top']
rad_int = variables['rad_int']
asp_rat_int = variables['asp_rat_int']
shap_int = variables['shap_int']
trans_int_top = variables['trans_int_top']
norm_vec_int_top = variables['norm_vec_int_top']
norm_vec_int_bot = variables['norm_vec_int_bot']
shap_frac_fam = variables['shap_frac_fam']
dist_frac_fam = variables['dist_frac_fam']
kappa_frac_fam = variables['kappa_frac_fam']
theta_frac_fam = variables['theta_frac_fam']
phi_frac_fam = variables['phi_frac_fam']
alpha_frac_fam = variables['alpha_frac_fam']
min_radius_frac_fam = variables['min_radius_frac_fam']
max_radius_frac_fam = variables['max_radius_frac_fam']
p32_frac_fam = variables['p32_frac_fam']
hy_variable_frac_fam = variables['hy_variable_frac_fam']
hy_function_frac_fam = variables['hy_function_frac_fam']
hy_params_frac_fam = variables['hy_params_frac_fam']
shap_1 = variables['shap_1']
radi_1 = variables['radi_1']
fault_center_x_1 = variables['fault_center_x_1']
fault_center_y_1 = variables['fault_center_y_1']
fault_center_z_1 = variables['fault_center_z_1']
trans_1 = variables['trans_1']
norm_vec_1 = variables['norm_vec_1']
ap_1 = variables['ap_1']
shap_2 = variables['shap_2']
radi_2 = variables['radi_2']
fault_center_x_2 = variables['fault_center_x_2']
fault_center_y_2 = variables['fault_center_y_2']
fault_center_z_2 = variables['fault_center_z_2']
trans_2 = variables['trans_2']
norm_vec_2 = variables['norm_vec_2']
ap_2 = variables['ap_2']
# ---------------------------------------

jobname = "combined_UDFM"
DFN = DFNWORKS(jobname,
               flow_solver="FEHM",
               ncpu=12)

DFN.params['domainSize']['value'] = [domain_size_x, domain_size_y, domain_size_z]
DFN.params['h']['value'] = h
DFN.params['disableFram']['value'] = True
DFN.params['keepIsolatedFractures']['value'] = True
DFN.params['seed']['value']=seed_value # create the same dfn each time

#exit()
# DUMMY FRACTURE FAMILY, needed to trick code 
DFN.add_fracture_family(shape=shap_frac_fam,
                        distribution=dist_frac_fam,
                        kappa=kappa_frac_fam,
                        theta=theta_frac_fam,
                        phi=phi_frac_fam,
                        alpha=alpha_frac_fam,
                        min_radius=min_radius_frac_fam,
                        max_radius=max_radius_frac_fam,
                        p32=p32_frac_fam,
                        hy_variable=hy_variable_frac_fam,
                        hy_function=hy_function_frac_fam,
                        hy_params=hy_params_frac_fam)


DFN.h = 1000 
DFN.x_min = -(domain_size_x/2)
DFN.y_min = -(domain_size_y/2)
DFN.z_min = -(domain_size_z/2)
DFN.x_max = (domain_size_x/2)
DFN.y_max = (domain_size_y/2)
DFN.z_max = (domain_size_z/2)

DFN.domain = {"x": domain_size_x, "y": domain_size_y, "z": domain_size_z}

src_dir = os.getcwd()


os.symlink('middle_layer/middle_layer.inp', 'middle_layer.inp')
#os.symlink('faults/faults.inp', 'faults.inp')


lagrit_script = """"
## prior to running you need to copy the reduced_mesh from the top & bottom DFN here (or symbolic link)
## also copy the *pkl files
# to run 
# lagrit < combine_mesh.lgi 

# read in mesh 1 
read / middle_layer.inp / mo_middle

# read in mesh 2 
#read / faults.inp / mo_faults /

# combine mesh 1 and mesh 2 to make final mesh
#addmesh / merge / mo_dfn / mo_middle / mo_faults

# write to file 
dump / combined_dfn.inp / mo_middle 
dump / reduced_mesh.inp / mo_middle 

finish 
"""

with open('combine_dfn.lgi', 'w') as fp:
    fp.write(lagrit_script)

import subprocess
subprocess.call('lagrit < combine_dfn.lgi', shell = True)

#exit() 

DFN.make_working_directory(delete=True)
DFN.check_input()

#FAULT_DFN = DFNWORKS(pickle_file = f"{src_dir}/faults/faults.pkl")
MIDDLE_DFN = DFNWORKS(pickle_file = f"{src_dir}/middle_layer/middle_layer.pkl")

## combine DFN
## combine DFN
DFN.num_frac = MIDDLE_DFN.num_frac 
DFN.centers = MIDDLE_DFN.centers
DFN.aperture = MIDDLE_DFN.aperture
DFN.perm = MIDDLE_DFN.perm
DFN.transmissivity =  MIDDLE_DFN.transmissivity

#_DFN.polygons.copy() 
DFN.polygons = MIDDLE_DFN.polygons
DFN.normal_vectors = MIDDLE_DFN.normal_vectors

os.symlink(f"{src_dir}/reduced_mesh.inp", "reduced_mesh.inp")

DFN.map_to_continuum(l = l, orl = orl)
DFN.upscale(mat_perm=mat_perm_middle, mat_por=mat_por_middle)

# load z values 
with open('octree_dfn.inp') as finp:
    header = finp.readline().split()
    num_nodes = int(header[0])
    print(num_nodes)
    x = np.zeros(num_nodes)
    y = np.zeros(num_nodes)
    z = np.zeros(num_nodes)
    for i in range(num_nodes):
        line = finp.readline().split()
        x[i] = float(line[1])
        y[i] = float(line[2])
        z[i] = float(line[3])
material_id = np.genfromtxt("tag_frac.dat").astype(int)

df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'material': material_id, })
print(df)
df.to_pickle('octree_nodes.pkl')

lagrit_script = """
read / octree_dfn.inp / mo1
cmo / addatt / mo1 / frac_index / vdouble / scalar / nnodes
cmo / setatt / mo1 / frac_index / 1 0 0 / 1
cmo / readatt / mo1 / frac_index / 1, 0, 0 / tag_frac.dat 
dump / tmp.inp / mo1 
finish
"""
with open("color_mesh.lgi", "w") as fp:
    fp.write(lagrit_script)

subprocess.call('lagrit < color_mesh.lgi', shell = True)

#DFN.correct_stor_file()
#os.symlink(dfnFlow_file, 'fehmn.files')
#DFN.fehm()


# DFN.zone2ex(zone_file='all')

# DFN.pflotran()
# DFN.parse_pflotran_vtk_python()
# DFN.pflotran_cleanup()

