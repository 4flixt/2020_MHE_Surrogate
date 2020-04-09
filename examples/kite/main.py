#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

""" User settings: """
show_animation = False
store_results = True

"""
Get configured do-mpc modules:
"""

model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)

"""
Set initial state
"""

theta_0 = 0.39359907+0.05
phi_0 = 0.72791537
psi_0 = 0.1
x0 = np.array([theta_0, phi_0, psi_0]).reshape(-1,1)

mpc.set_initial_state(x0, reset_history=True)
simulator.set_initial_state(x0, reset_history=True)
estimator.set_initial_state(x0, reset_history=True)

"""
Setup graphic:
"""
fig, ax = plt.subplots(figsize=(8,5))
color = plt.rcParams['axes.prop_cycle'].by_key()['color']

phi_pred = mpc.data.prediction(('_x', 'phi'))[0]
theta_pred = mpc.data.prediction(('_x', 'theta'))[0]
pred_lines = ax.plot(phi_pred, theta_pred, color=color[0], linestyle='--', linewidth=1)

phi = mpc.data['_x', 'phi']
theta = mpc.data['_x', 'theta']

res_lines = ax.plot(phi, theta, color=color[0])


#fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data, figsize=(8,5))

# graphics = do_mpc.graphics.Graphics(simulator.data)
#
# graphics.add_line(x = ('_x', 'phi'), y=('_x','theta'), axis=ax)
#
plt.ion()

"""
Run MPC main loop:
"""

for k in range(500):
    u0 = mpc.make_step(x0)
    for m in range(3):
        y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)

    if show_animation:
        phi_pred = mpc.data.prediction(('_x', 'phi'))[0]
        theta_pred = mpc.data.prediction(('_x', 'theta'))[0]
        for i in range(phi_pred.shape[1]):
            pred_lines[i].set_data(phi_pred[:,i], theta_pred[:,i])
        phi = mpc.data['_x', 'phi']
        theta = mpc.data['_x', 'theta']
        res_lines[0].set_data(phi, theta)
        ax.relim()
        ax.autoscale()

        # graphics.plot_results()
        # #graphics.plot_predictions()
        # graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([mpc, simulator], 'kite')
