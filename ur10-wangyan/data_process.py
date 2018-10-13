# 1 - Praparing work
# Import libraries
from sympy import init_printing
init_printing()
    
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 8]

import numpy as np
import sympy
from robot_def import RobotDef
from kinematics import Geometry
from dynamics import Dynamics
from trajectory_optimization import TrajOptimizer
from trajectory_optimization import TrajPlotter
from utils import new_sym
from utils import save_data, load_data
import time


q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10 = new_sym('q:11')
_pi = sympy.pi

dh = []
springs = []
friction_type = []
tendon_couplings = []



dh = [(0,   -1, [1],    0,      0,      0,           0,         False, False, False),
      (1,   0,  [2],    0,      0,      0.128,       q1,        True,  False, False),
      (2,   1,  [3],    0,      -_pi/2, 0.176,    q2-_pi/2,     True,  False, False),
      (3,   2,  [4],    0.612,  0,      -0.128,      q3,        True,  False, False),
      (4,   3,  [5],    0.572,  0,      0.116,      q4+_pi/2,   True,  False, False),
      (5,   4,  [6],    0,    _pi/2,    0.116,       q5,        True,  False, False),
      (6,   5,  [],     0,    -_pi/2,   0.092,       q6,        True,  False, False)]
    
friction_type = []
robot_def = RobotDef(dh,
                     springs=springs,
                     tendon_couplings=tendon_couplings,
                     dh_convention='mdh',
                     friction_type=friction_type)





# Data processing
print("\n\n")
print("Data processing---------------------------")
from identification import load_trajectory_data, diff_and_filt_data, plot_trajectory_data, plot_meas_pred_tau, gen_regressor



# Load data
# Load robot model
model_name = 'ur10'
training_trajectory_name = 'ur10_0'


model_folder = 'data/' + model_name + '/model/'
robot_model = load_data(model_folder,model_name)


trajectory_folder = 'data/' + model_name +'/optimal_trajectory/'
dof, fourier_order, base_freq, traj_optimizer_result, reg_norm_mat = load_data(trajectory_folder,                            
                                                                               training_trajectory_name)

# dof = 6
# fourier_order = 6
# base_freq = 0.1

print("dof: {}".format(dof))
print("Fourier order: {}".format(fourier_order))
print("Base frequency: {}".format(base_freq))

print("robot_model.coordinates: ")
print(robot_model.coordinates)

# Load traning data set
# training_trajectory_name = 'one'

results_folder = 'data/' + model_name +'/measured_trajectory/'
training_results_data_file = results_folder + training_trajectory_name + '_results.csv'

trajectory_sampling_rate = 50
t_train, q_raw_train, dq_raw_train, tau_raw_train = load_trajectory_data(training_results_data_file,
                                                                   trajectory_sampling_rate)


# Calculate filter cut-off frequency
# times of the highest frequency in the Fourier series
fc_mult = 10.0 #4.0

fc = base_freq * fourier_order * fc_mult
print("Cut frequency: {}".format(fc))


# Differentiation and filtering
# Traning data
t_cut_train, q_f_train, dq_f_train, ddq_f_train, tau_f_train, q_raw_cut_train, tau_raw_cut_train =\
    diff_and_filt_data(dof, 1.0/trajectory_sampling_rate,
                       t_train, q_raw_train, dq_raw_train, tau_raw_train, fc, fc, fc, fc)
# plot_trajectory_data(t_cut_train, q_raw_cut_train, q_f_train, dq_f_train, ddq_f_train,
#                      tau_raw_cut_train, tau_f_train)



# Regression

# Ordinary Least Square (OLS)
print("\n\n")
print("Ordinary Least Square (OLS)----------------------------")
# Generate regressor matrix for base parameters
base_param_num = robot_model.base_num
H_b_func = robot_model.H_b_func
W_b_train, tau_s_train = gen_regressor(base_param_num, H_b_func, q_f_train,
                                       dq_f_train, ddq_f_train, tau_f_train)


W_b_train.shape, tau_s_train.shape


xb_ols = np.linalg.lstsq(W_b_train, tau_s_train)[0]

table = [["Base Parameter", "Value"]]

for i in range(robot_model.base_num):
    param_str = str(sympy.Matrix(robot_model.base_param)[i])
    max_disp_len = 50
    line = [param_str if len(param_str) <= max_disp_len 
            else param_str[:max_disp_len] + '...', xb_ols[i]]
    #['%.7s ...'%b if len(str(b)) > 7 else str(b)
    table.append(line)

print(table)

# Compare measured torque and predicted torque on the training set
# Generate predicted torque
tau_p_train = np.zeros(tau_f_train.shape)
tau_ps_train = W_b_train.dot(xb_ols)
for i in range(dof):
    tau_p_train[:, i] = tau_ps_train[i::dof]

# Evaluate regression
var_regression_error_ols = np.linalg.norm(tau_ps_train - tau_s_train) / \
                        (tau_ps_train.size - base_param_num)
print("variance of regression error using OLS:")
print(var_regression_error_ols)

std_dev_xb_ols = np.sqrt(np.diag(var_regression_error_ols *
                                 np.linalg.inv(W_b_train.transpose().dot(W_b_train))))
print("standard deviation of xb using OLS:")
print(std_dev_xb_ols)

pct_std_dev_xb_ols = std_dev_xb_ols / np.abs(xb_ols)
print("percentage of standard deviation of xb using OLS: ")
print(pct_std_dev_xb_ols)


# Plot measured torque and predicted torque

plot_data_num = int(1 / base_freq * trajectory_sampling_rate)
# plot_meas_pred_tau(t_cut_train[:plot_data_num], tau_f_train[:plot_data_num, :],
#                    tau_p_train[:plot_data_num, :], robot_def.coordinates_joint_type)





# Weighted Least Square (WLS)
print("\n\n")
print("Weighted Least Square (WLS)--------------------------------")
# Training data set
# weight = np.sqrt(np.linalg.norm(tau_f - tau_p, axis=0)/(tau_f.shape[0] - base_param_num))
weight = np.max(tau_f_train, axis=0) - np.min(tau_f_train, axis=0)
weight
weight[1] /= 1
weight[2] /= 1
weight[3] /= 1
print("weight: ")
print(weight)

print("W_b_train.shape, tau_s_train.shape: ")
print(W_b_train.shape, tau_s_train.shape)

# repeat the weight to generate a large vecoter for all the data
weights = 1.0/np.tile(weight, W_b_train.shape[0]/weight.shape[0])

print("weights.shape: ")
print(weights.shape)

W_b_wls_train = np.multiply(W_b_train, np.asmatrix(weights).transpose())
tau_s_wls_train = np.multiply(tau_s_train, weights)

xb_wls = np.linalg.lstsq(W_b_wls_train, tau_s_wls_train)[0]

#np.set_printoptions(precision=2)
print("xb_wls: ")
print(sympy.Matrix.hstack(sympy.Matrix(robot_model.base_param), sympy.Matrix(xb_wls)))

tau_p_wls_train = np.zeros(tau_f_train.shape)
tau_ps_wls_train = W_b_train.dot(xb_wls)
for i in range(dof):
    tau_p_wls_train[:, i] = tau_ps_wls_train[i::dof]

# plot_meas_pred_tau(t_cut_train[:plot_data_num], tau_f_train[:plot_data_num, :],
#                    tau_p_wls_train[:plot_data_num, :], robot_def.coordinates_joint_type)    

print("norm: ")
print(np.linalg.norm(tau_f_train[:plot_data_num, :] - tau_p_wls_train[:plot_data_num, :], axis=0)\
    / np.linalg.norm(tau_f_train[:plot_data_num, :], axis=0))


print("robot_model.bary_param: ")
print(robot_model.bary_param)



# Convex optimization
print("\n\n")
print("Convex optimization:---------------------------")
# Generate regressor matrix for barycentric parameters
from identification.sdp_opt import SDPOpt

bary_param_num = len(robot_model.bary_param)
H_func = robot_model.H_func
W_train, tau_s_train = gen_regressor(bary_param_num, H_func,
                                     q_f_train, dq_f_train, ddq_f_train, tau_f_train)


W_w_train = np.multiply(W_train, np.asmatrix(weights).transpose())
tau_w_s_train = np.multiply(tau_s_train, weights)

print("len(robot_model.std_param): ")
print(len(robot_model.std_param))


sdp_constraints = []
spring_constraints = []
# sdp_constraints = [   (0.5, 10, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 0.5, 0.5, 0.5),
#                       (0.5, 10, -0.1, 0.7, -0.1, 0.1, -0.1, 0.1, 0.5, 0.5, 0.5),
#                       (0.5, 10, -0.1, 0.6, -0.1, 0.1, -0.1, 0.1, 0.1, 0.5, 0.5),
#                       (0.5, 10, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 0.1, 0.1, 0.1),
#                       (0.5, 10, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 0.1, 0.1, 0.1),
#                       (0.5, 10, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, 0.1, 0.1, 0.1)]

sdp_constraints = [   (0.1, 10, -1.5, 1.5, -1.5, 1.5, -1.5, 1.5, 1.5, 1.5, 1.5),
                      (0.1, 10, -1.5, 1.5, -1.5, 1.5, -1.5, 1.5, 1.5, 1.5, 1.5),
                      (0.1, 10, -1.5, 1.5, -1.5, 1.5, -1.5, 1.5, 1.5, 1.5, 1.5),
                      (0.1, 10, -1.5, 1.5, -1.5, 1.5, -1.5, 1.5, 0.1, 0.1, 0.1),
                      (0.1, 10, -1.5, 1.5, -1.5, 1.5, -1.5, 1.5, 0.1, 0.1, 0.1),
                      (0.1, 10, -1.5, 1.5, -1.5, 1.5, -1.5, 1.5, 0.1, 0.1, 0.1)]


sdp_opt_std = SDPOpt(W_w_train, tau_w_s_train, robot_model,
                     sdp_constraints, spring_constraints)
# sdp_opt_std = SDPOpt(W, tau_s, robot_model, sdp_constraints)
sdp_opt_std.solve()

# Compare measured torque and predicted torque on training set
# Generate predicted torque

tau_p_sdp_train = np.zeros(tau_f_train.shape)
tau_ps_sdp_train = W_train.dot(sdp_opt_std.x_result)
for i in range(dof):
    tau_p_sdp_train[:, i] = tau_ps_sdp_train[i::dof]

print("norm: ")
print(np.linalg.norm(tau_f_train[:plot_data_num, :] - tau_p_sdp_train[:plot_data_num, :], axis=0)\
    / np.linalg.norm(tau_f_train[:plot_data_num, :], axis=0))

# Plot measured torque and predicted torque
plot_data_num = int(1 / base_freq * trajectory_sampling_rate)
plot_meas_pred_tau(t_cut_train[:plot_data_num]- t_cut_train[0],
                   tau_f_train[:plot_data_num, :],
                   tau_p_sdp_train[:plot_data_num, :],
                   robot_def.coordinates_joint_type)

print("bary_param x_result: ")
print(sympy.Matrix.hstack(sympy.Matrix(robot_model.bary_param), sympy.Matrix(sdp_opt_std.x_result)))

from identification import barycentric2standard_params

x_std = barycentric2standard_params(sdp_opt_std.x_result, robot_model)

print("std_param x_result: ")
print(sympy.Matrix.hstack(sympy.Matrix(robot_model.std_param), sympy.Matrix(x_std)))

print(x_std)
