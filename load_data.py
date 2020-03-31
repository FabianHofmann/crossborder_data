# -*- coding: utf-8 -*-
"""
Created on Mon May 20 20:07:12 2019

@author: Hazem
"""

import os
import pandas as pd
import auxiliary_functions as aux
import logging

base_path = os.path.dirname(os.path.realpath(__file__))
directory = os.path.join(base_path, 'data', 'TP')
filenames = os.listdir(directory)

if not os.path.isdir('processed'):
    os.mkdir('processed')

def load_from_file(fn):
    """
    Load and standardize the raw data file from directory data/TP.

    The directory should contain all necessary raw data files parsed from the
    entsoe transparency website.

    Parameters
    ----------
    fn : str
        filename

    """
    #Note: some areas that are not countries as: "Italy_Sacodc" do not contain
    #any of the words in selection but are still removed because they're if
    #they're in the InAreaName they'll always be paired with an area containing
    #one of the selection in the OutAreaName and vice versa.
    logging.info(f'Loading data from {fn}')
    path = os.path.join(directory, fn)
    df = pd.read_csv(path, encoding = 'UTF-16', sep="\t",\
                     parse_dates=['DateTime'], index_col='DateTime')

    #Remove all area names containing the words in selection
    selection = [' BZ', ' CA', ' CTA', ' AC']
    for element in selection:
        df = df[~df['InAreaName'].str.contains(element)]
        df = df[~df['OutAreaName'].str.contains(element)]

    #Dropping unnescessary columns
    return df.iloc[:,7:]\
             .drop(['InAreaCode', 'InAreaTypeCode', 'InAreaName', 'UpdateTime'],
                   axis=1)\
             .sort_index()

df = pd.concat([load_from_file(fn) for fn in filenames]).sort_index()

rawcbf =  df.reset_index()\
            .pivot_table(index='DateTime',
                         columns=['OutMapCode','InMapCode'],
                         values='FlowValue')

cbf = aux.directed_cbf(rawcbf)
cbf.to_csv(os.path.join('processed', 'cbf_loaded.csv'))

