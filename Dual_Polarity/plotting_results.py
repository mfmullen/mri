#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 08:32:44 2021

@author: mmullen
"""

import numpy as np
import plotly.express as px
import plotly.io as pio
pio.renderers.default='png'

if ('Ic' not in locals()) or ('Tx' not in locals()):

    Ic = np.load('20201209_3T_data_rsos_full_processed.npz')['Ic']
    Tx = np.load('20201209_3T_data_rsos_full_processed.npz')['Tx']

fig = px.imshow(Tx[:,128,:], color_continuous_scale='jet',aspect='equal')
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.update_layout(coloraxis_colorbar=dict(
    title="Shift"))
fig.show()

fig2 = px.imshow(Ic[:,128,:],color_continuous_scale='gray',aspect='equal')
fig2.update_layout(coloraxis_showscale=False)
fig2.update_xaxes(showticklabels=False)
fig2.update_yaxes(showticklabels=False)
fig2.show()
