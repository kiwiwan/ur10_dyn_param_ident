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


# Folder dir for saving and loading files
model_name = 'ur10'
model_folder = 'data/' + model_name + '/model/'



# 2 - Robot modelling
# Robot geometry definition in following order

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



robot_def.dq_for_frame
robot_def.coordinates_joint_type
robot_def.bary_params


# Create kinematics chain
geom = Geometry(robot_def)
from numpy import deg2rad
angle = [0, deg2rad(-90), 0, deg2rad(90), 0, 0]
geom.draw_geom(angle)


# Create dynamics
dyn = Dynamics(robot_def, geom)
robot_def.bary_params
sympy.Matrix(dyn.base_param)




from data import RobotModel
# Data to be saved
robot_model = RobotModel(dyn)
# Save
save_data(model_folder, model_name, robot_model)
print('Saved {} parameters'.format(len(robot_model.base_param)))

