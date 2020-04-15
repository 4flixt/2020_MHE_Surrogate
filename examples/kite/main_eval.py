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
from tensorflow import keras
import pdb
import sys
sys.path.append('../../')
import do_mpc
import pickle
from RNN_tools import get_model

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

""" User settings: """
show_animation = True
user_anim = True
store_results = False

"""
Get configured do-mpc modules:
"""
w_ref = 10+5*np.random.rand()
E_0 = 6+np.random.rand()

model = template_model(state_feedback = False)
mpc = template_mpc(model, w_ref, E_0)
simulator = template_simulator(model, w_ref, E_0)

"""
LSTM estimator
"""
rnn_model = '006'

with open('./models/{model}_RNN/{model}_RNN_config.pkl'.format(model=rnn_model),'rb') as f:
    model_config = pickle.load(f)

rnn_estimator = get_model(model_config['model_param'], nx=3, ny=3, batch_size=1, seq_length=1, stateful=True)
rnn_estimator.load_weights('./models/{model}_RNN/{model}_RNN_weights'.format(model=rnn_model), by_name=False)

def estimator_update(rnn_in):
    rnn_in_scaled = rnn_in.reshape(1,-1)/model_config['x_scaling']
    rnn_out_scaled = rnn_estimator.predict(rnn_in_scaled.reshape(1,1,-1)).squeeze()
    rnn_out = rnn_out_scaled*model_config['y_scaling']
    return rnn_out



"""
Set initial state
"""
# Derive initial state from bounds:
lb_theta, ub_theta = mpc.bounds['lower','_x','theta'], mpc.bounds['upper','_x','theta']
lb_phi, ub_phi = mpc.bounds['lower','_x','phi'], mpc.bounds['upper','_x','phi']
lb_psi, ub_psi = mpc.bounds['lower','_x','psi'], mpc.bounds['upper','_x','psi']
# with mean and radius:
m_theta, r_theta = (ub_theta+lb_theta)/2, (ub_theta-lb_theta)/2
m_phi, r_phi = (ub_phi+lb_phi)/2, (ub_phi-lb_phi)/2
m_psi, r_psi = (ub_psi+lb_psi)/2, (ub_psi-lb_psi)/2
# How close can the intial state be to the bounds?
# tightness=1 -> Initial state could be on the bounds.
# tightness=0 -> Initial state will be at the center of the feasible range.
tightness = 0.5
theta_0 = m_theta-tightness*r_theta+2*tightness*r_theta*np.random.rand()
phi_0 = m_phi-tightness*r_phi+2*tightness*r_phi*np.random.rand()
psi_0 = m_psi-tightness*r_psi+2*tightness*r_psi*np.random.rand()

x0 = np.array([theta_0, phi_0, psi_0]).reshape(-1,1)

mpc.set_initial_state(x0, reset_history=True)
simulator.set_initial_state(x0, reset_history=True)

"""
Setup graphic:
"""
if user_anim:
    fig, ax = plt.subplots(figsize=(8,5))
    color = plt.rcParams['axes.prop_cycle'].by_key()['color']

    phi_pred = mpc.data.prediction(('_x', 'phi'))[0]
    theta_pred = mpc.data.prediction(('_x', 'theta'))[0]
    pred_lines = ax.plot(phi_pred, theta_pred, color=color[0], linestyle='--', linewidth=1)

    phi = mpc.data['_x', 'phi']
    theta = mpc.data['_x', 'theta']

    res_lines = ax.plot(phi, theta, color=color[0])
else:
    fig, ax, graphics = do_mpc.graphics.default_plot(simulator.data)


plt.ion()

"""
Run MPC main loop:
"""

for k in range(700):
    u0 = mpc.make_step(x0)

    y_next = simulator.make_step(u0)

    rnn_out = estimator_update(y_next)
    x0 = np.array([rnn_out[1], rnn_out[0], rnn_out[2]])
    #x0 = estimator.make_step(y_next)

    if show_animation:
        if user_anim:
            phi_pred = mpc.data.prediction(('_x', 'phi'))[0]
            theta_pred = mpc.data.prediction(('_x', 'theta'))[0]
            for i in range(phi_pred.shape[1]):
                pred_lines[i].set_data(phi_pred[:,i], theta_pred[:,i])
            phi = mpc.data['_x', 'phi']
            theta = mpc.data['_x', 'theta']
            res_lines[0].set_data(phi, theta)
            ax.relim()
            ax.autoscale()
        else:
            graphics.plot_results()
            #graphics.plot_predictions()
            graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([simulator], 'kite')
