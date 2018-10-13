from sympy import init_printing
init_printing()
    
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [24, 16]#[12, 8]

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

model_name = 'ur10'
from utils import load_data
model_folder = 'data/' + model_name + '/model/'
robot_model = load_data(model_folder, model_name)

trajectory_name = 'ur10'


from numpy import deg2rad

optimal_trajectory_folder = 'data/optimal_trajectory/'
trajectory_folder = 'data/' + model_name +'/optimal_trajectory/'

base_freq = 0.05
fourier_order = 6

joint_constraints = []
cartesian_constraints = []

joint_constraints = [(q1, deg2rad(-180), deg2rad(180), deg2rad(-120), deg2rad(120)),
                         (q2, deg2rad(-180), deg2rad(180), deg2rad(-120), deg2rad(120)),
                         (q3, deg2rad(-180), deg2rad(180), deg2rad(-120), deg2rad(120)),
                         (q4, deg2rad(-180), deg2rad(180), deg2rad(-120), deg2rad(120)),
                         (q5, deg2rad(-180), deg2rad(180), deg2rad(-120), deg2rad(120)),
                         (q6, deg2rad(-180), deg2rad(180), deg2rad(-120), deg2rad(120))]


traj_optimizer = TrajOptimizer(robot_model, fourier_order, base_freq,
                               joint_constraints=joint_constraints,
                               cartesian_constraints = cartesian_constraints)
                                                                                                 
                                                                                                                     
traj_optimizer.optimize()


reg_norm_mat = traj_optimizer.calc_normalize_mat()

traj_optimizer.calc_frame_traj()

traj_plotter = TrajPlotter(traj_optimizer.fourier_traj,traj_optimizer.frame_traj,
                           traj_optimizer.const_frame_ind, robot_model.coordinates)
traj_plotter.plot_desired_traj(traj_optimizer.x_result)

traj_plotter.plot_frame_traj(True)


dof_order_bf_x_norm = (traj_optimizer.fourier_traj.dof, fourier_order,
                       base_freq, traj_optimizer.x_result, reg_norm_mat)
save_data(trajectory_folder, trajectory_name, dof_order_bf_x_norm)

freq = 50
tf = 65 # including 5 seconds' prestable time
traj_optimizer.make_traj_csv(trajectory_folder, trajectory_name, freq, tf)








