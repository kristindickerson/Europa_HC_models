#"""
#   :synopsis: Create yaml
#   :version: 1.0
#   :maintainer: Kristin Dickerson
#.. moduleauthor:: Kristin Dickerson <krdicker@ucsc.edu>
#"""

import yaml

## STATIC 
# ---------------------------------------
# octree refinement data
l = 2000
orl = 2

# file information
flow_solver_in = "FEHM"
ncpu_in        = 8
seed_value     = 1

# domain size
domain_size_x = 100000.0
domain_size_y = 100000.0
domain_size_z = 50000.0

# domain layering
s_z = 5000
aq_z = 10000
cb_z = 35000
tot_z = s_z + aq_z + cb_z

# layer matrix permeabilities
mat_perm_top    = 1.0e-15
mat_perm_middle = 1.0e-15
mat_perm_bottom = 1.0e-15
mat_por_top    = 0.01
mat_por_middle = 0.01
mat_por_bottom = 0.01

# middle layer
midlayer_size_x = domain_size_x
midlayer_size_y = domain_size_y
midlayer_size_z = aq_z

midlayer_center_x = 0
midlayer_center_y = 0
midlayer_center_z = ((tot_z/2) - (s_z + aq_z)) + (aq_z/2)

h = 100

# middle layer interfaces
interface_bot = -(aq_z/2)
interface_top = (aq_z/2)
rad_int = domain_size_y/2
asp_rat_int = 1.0
shap_int = 'rect'
rad_int = domain_size_y/2
trans_int_top = [0, 0, interface_top]
trans_int_bot = [0, 0, interface_bot]
norm_vec_int_top = [0, 0, 1]
norm_vec_int_bot = [0, 0, 1]

# middle layer small fractures
shap_frac_fam="ell"
dist_frac_fam="tpl"
kappa_frac_fam=0.1
theta_frac_fam=0.0
phi_frac_fam=0.0
alpha_frac_fam=1.2
min_radius_frac_fam=100.0
max_radius_frac_fam=1000.0
p32_frac_fam=.0000001
hy_variable_frac_fam="aperture"
hy_function_frac_fam="constant"
hy_params_frac_fam={"mu": 1e-4}
                        
                        
## fault 1
shap_1 = 'rect'
radi_1 = (aq_z+s_z)/2

fault_center_x_1 = 0
fault_center_y_1 = -15000 #-(midlayer_size_y/2)/2
fault_center_z_1 = (aq_z+s_z)/6

trans_1 = [fault_center_x_1, fault_center_y_1, fault_center_z_1]
norm_vec_1 = [0, 1, 0] # this is vertical
ap_1 = 1.0e-4 

## fault 2
shap_2 = 'rect'
radi_2 = (aq_z+s_z)/2

fault_center_x_2 = 0
fault_center_y_2 = 15000 #(midlayer_size_y/2)/2
fault_center_z_2 = (aq_z+s_z)/6

trans_2 = [fault_center_x_2, fault_center_y_2, fault_center_z_2]
norm_vec_2 = [0, 1, 0] # this is vertical
ap_2 = 1.0e-4 

## fault 3
shap_3 = 'rect'
radi_3 = (aq_z+s_z)/2

fault_center_x_3 = 0
fault_center_y_3 = 0
fault_center_z_3 = (aq_z+s_z)/6

trans_3 = [fault_center_x_3, fault_center_y_3, fault_center_z_3]
norm_vec_3 = [0, 1, 0] # this is vertical
ap_3 = 1.0e-4 

# Create a dictionary with both static and calculated values
variables = {
    'l': l,
    'orl': orl,
    'flow_solver_in': flow_solver_in,
    'ncpu_in': ncpu_in,
    'seed_value': seed_value,
    's_z': s_z,
    'aq_z': aq_z,
    'cb_z': cb_z,
    'tot_z': tot_z,
    'domain_size_x': domain_size_x,
    'domain_size_y': domain_size_y,  
    'domain_size_z': domain_size_z,
    'mat_perm_top': mat_perm_top,
    'mat_perm_middle': mat_perm_middle,
    'mat_perm_bottom': mat_perm_bottom,
    'mat_por_top': mat_por_top,
    'mat_por_middle': mat_por_middle,
    'mat_por_bottom': mat_por_bottom,
    'midlayer_size_x': midlayer_size_x,
    'midlayer_size_y': midlayer_size_y,
    'midlayer_size_z': midlayer_size_z,
    'h': h,
    'midlayer_center_x': midlayer_center_x,
    'midlayer_center_y': midlayer_center_y,
    'midlayer_center_z': midlayer_center_z,  
    'interface_bot': interface_bot,
    'interface_top': interface_top,
    'rad_int': rad_int,
    'asp_rat_int': asp_rat_int,   
    'shap_int': shap_int,
    'rad_int': rad_int,
    'trans_int_top': trans_int_top,
    'trans_int_bot': trans_int_bot,
    'norm_vec_int_top': norm_vec_int_top,
    'norm_vec_int_bot': norm_vec_int_bot,
    'shap_frac_fam': shap_frac_fam,
    'dist_frac_fam': dist_frac_fam,
    'kappa_frac_fam': kappa_frac_fam,
    'theta_frac_fam': theta_frac_fam,
    'phi_frac_fam': phi_frac_fam,
    'alpha_frac_fam': alpha_frac_fam,
    'min_radius_frac_fam': min_radius_frac_fam,
    'max_radius_frac_fam': max_radius_frac_fam,
    'p32_frac_fam': p32_frac_fam,
    'hy_variable_frac_fam': hy_variable_frac_fam,
    'hy_function_frac_fam': hy_function_frac_fam,
    'hy_params_frac_fam': hy_params_frac_fam,
    'shap_1': shap_1,
    'radi_1': radi_1,
    'fault_center_x_1': fault_center_x_1,
    'fault_center_y_1': fault_center_y_1,
    'fault_center_z_1': fault_center_z_1,
    'trans_1': trans_1,  
    'norm_vec_1': norm_vec_1,
    'ap_1': ap_1,
    'shap_2': shap_2,            
    'radi_2': radi_2,    
    'fault_center_x_2': fault_center_x_2,
    'fault_center_y_2': fault_center_y_2,
    'fault_center_z_2': fault_center_z_2,
    'trans_2': trans_2,
    'norm_vec_2': norm_vec_2,
    'ap_2': ap_2, 
    'shap_3': shap_3,            
    'radi_3': radi_3,    
    'fault_center_x_3': fault_center_x_3,
    'fault_center_y_3': fault_center_y_3,
    'fault_center_z_3': fault_center_z_3,
    'trans_3': trans_3,
    'norm_vec_3': norm_vec_3,
    'ap_3': ap_3  
}

# Create the final config structure
config = {
    'variables': variables
}

# Write the config dictionary to a YAML file
with open('config.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)

print("YAML file 'config.yaml' with calculated values created successfully.")