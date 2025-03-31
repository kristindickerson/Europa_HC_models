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
l = variables['l']
orl = variables['orl']
flow_solver_in = variables['flow_solver_in']
ncpu_in = variables['ncpu_in']
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
shap_3 = variables['shap_3']
radi_3 = variables['radi_3']
fault_center_x_3 = variables['fault_center_x_3']
fault_center_y_3 = variables['fault_center_y_3']
fault_center_z_3 = variables['fault_center_z_3']
trans_3 = variables['trans_3']
norm_vec_3 = variables['norm_vec_3']
ap_3 = variables['ap_3']
# ---------------------------------------

src_path = os.getcwd()
jobname = src_path + "/faults"
dfnFlow_file = os.getcwd() + '/fehmn.files'

DFN = DFNWORKS(jobname,
               dfnFlow_file=dfnFlow_file,
               flow_solver=flow_solver_in,
               ncpu=ncpu_in)

DFN.params['domainSize']['value'] = [domain_size_x, domain_size_y, domain_size_z]
DFN.params['h']['value'] = h
DFN.params['disableFram']['value'] = True
DFN.params['keepIsolatedFractures']['value'] = True
DFN.params['domainCenter']['value'] = [midlayer_center_x, midlayer_center_y, midlayer_center_z]

#Individual fractures
DFN.add_user_fract(shape=shap_1,
                   radii=radi_1,
                   translation=trans_1,
                   normal_vector=norm_vec_1,
                   aperture=ap_1)

DFN.add_user_fract(shape=shap_2,
                   radii=radi_2,
                   translation=trans_2,
                   normal_vector=norm_vec_2,
                   aperture=ap_2)

DFN.add_user_fract(shape=shap_3,
                   radii=radi_3,
                   translation=trans_3,
                   normal_vector=norm_vec_3,
                   aperture=ap_3)


DFN.make_working_directory(delete=True)
DFN.check_input()
DFN.create_network()
DFN.mesh_network()
DFN.to_pickle("faults")
os.rename('reduced_mesh.inp', 'faults.inp')
exit() 
DFN.map_to_continuum(l=l, orl=orl)
exit()
DFN.upscale(mat_perm=mat_perm, mat_por=0.01)


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

exit()
####
lagrit_script = """
## read in dfn 
read / reduced_mesh.inp / mo_dfn
# make element set named elt_fracs s.t. itetclr (matid <= 3)
eltset / elt_fracs /  itetclr / lt / 3 
## remove the layer fractures
rmpoint / element / eltset, get, elt_fracs 
# read in the octree mesh 
read / octree_dfn.inp / mo_udfm


cmo / addatt / mo_udfm / tag / vdouble / scalar / nnodes
cmo / setatt / mo_udfm / tag / 1 0 0 / 1
cmo / readatt / mo_udfm / tag / 1, 0, 0 / tag_frac.dat


intersect_elements / mo_udfm / mo_dfn / fractures
eltset / eltfrac / fractures / ge 1 
pset / pfrac /  eltset / eltfrac 

pset/ p1 / geom / xyz / 1, 0, 0 /  & 
    -6000 -6000 -5000  /  6000 6000 1000 /  0,0,0

pset/ p2 / geom / xyz / 1, 0, 0 /  & 
    -6000 -6000 1000  /  6000 6000 4000 /  0,0,0

pset/ p3 / geom / xyz / 1, 0, 0 /  & 
    -6000 -6000 4000  /  6000 6000 6000 /  0,0,0

cmo / addatt / mo_udfm / layer / vdouble / scalar / nnodes
cmo / setatt / mo_udfm / layer / 1 0 0 / 0
cmo / setatt / mo_udfm / layer / pset, get, p1 / 1
cmo / setatt / mo_udfm / layer / pset, get, p2 / 2
cmo / setatt / mo_udfm / layer / pset, get, p3 / 3

dump / tmp.inp / mo_udfm 

finish 
"""






with open("paint_layers.lgi", "w") as fp:
    fp.write(lagrit_script)
    fp.flush() 


subprocess.call('lagrit < paint_layers.lgi' ,shell = True)

exit()



####

material_id = np.genfromtxt("tag_frac.dat").astype(int)
num_nodes = len(material_id)
perm = np.genfromtxt("perm_fehm.dat", skip_header = 1)[:,-1]
print(perm)
exit()

porosity = np.zeros(num_nodes)

# load z values 
with open('full_mesh.uge') as fuge:
    header = fuge.readline().split()
    num_cells = int(header[1])
    print(num_cells)
    z = np.zeros(num_cells)
    for i in range(num_cells):
        line = fuge.readline().split()
        z[i] = float(line[-2])

layer_1_matrix_perm = 1e-16 
layer_2_matrix_perm = 1e-12 
layer_3_matrix_perm = 1e-16

fracture_perm = 1e-10

for i in range(num_cells):
    if material_id[i] == 1: ## Matrix Cell
        if  z[i] < 1000:
            perm[i] = layer_1_matrix_perm
        elif 1000 < z[i] < 4000:
            perm[i] = layer_2_matrix_perm
        elif 4000 < z[i]:
            perm[i] = layer_3_matrix_perm

    # elif material_id[i] == 2:
    #     perm[i] = fracture_perm

with open("perm_layer.dat", "a") as fperm:
    fperm.write("perm\n")

for i in range(1, num_nodes + 1):
    with open("perm_layer.dat", "a") as fperm:
        fperm.write(
            str(i) + " " + str(i) + " " + "1" + " " +
            str(perm[i - 1]) + " " + str(perm[i - 1]) + " " +
            str(perm[i - 1]) + "\n")

with open("perm_layer.dat", "a") as fperm:
    fperm.write("")


with open("rock_fehm.dat", "a") as g:
    g.write(
        str(i) + " " + str(i) + " " + "1" + " " + "2165." + " " +
        "931." + " " + str(por_var[i - 1]) + "\n")



np.savetxt("my_perm.dat", perm)
lagrit_script = """
read / octree_dfn.inp / mo1
cmo / addatt / mo1 / perm / vdouble / scalar / nnodes
cmo / setatt / mo1 / perm / 1 0 0 / 1
cmo / readatt / mo1 / perm / 1, 0, 0 / my_perm.dat 
dump / perm.inp / mo1 
finish
"""
with open("color_mesh.lgi", "w") as fp:
    fp.write(lagrit_script)


subprocess.call('lagrit < color_mesh.lgi', shell = True)      
        
os.symlink(DFN.jobname + os.sep + "dfnGen_output", 'dfnGen_output')


# DFN.correct_stor_file()
# os.symlink(dfnFlow_file, 'fehmn.files')
# DFN.fehm()


# DFN.zone2ex(zone_file='all')

# DFN.pflotran()
# DFN.parse_pflotran_vtk_python()
# DFN.pflotran_cleanup()
