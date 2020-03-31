#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:57:58 2019

@author: fabian
"""

import pandas as pd
import auxiliary_functions as aux
import logging
logger = logging.getLogger(__name__)

# most nan are in 2016, if we drop this we have to filter out less
cbf_loaded = pd.read_csv('processed/cbf_loaded.csv', index_col=0,
                         parse_dates=True).loc['2017':'2018']

# 1. filter data
threshold=200
alpha=1.5
logger.info(f'Filtering: set alpha = {alpha} and threshold = {threshold}')
cbf_filtered = aux.filter_data(cbf_loaded, alpha=alpha, threshold=threshold,
                               per_year=True)
cbf_filtered.to_csv('intermediate_data/cbf_filtered.csv')

# 2. interpolate data
bc_average_week = aux.transform_to_average_week(cbf_filtered)
cbf_interpolated = cbf_filtered.fillna(bc_average_week)


# 3. scale data
cbf_processed = aux.scale_data(cbf_interpolated)
cbf_processed.to_csv('processed/cbf_processed.csv')
