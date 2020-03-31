# -*- coding: utf-8 -*-
"""
Created on Mon May 20 20:07:12 2019

@author: Hazem
"""

import os
import pandas as pd
import auxiliary_functions as aux

base_path = os.path.dirname(os.path.realpath(__file__))
directory = os.path.join(base_path, 'data', 'TP')
filenames = os.listdir(directory)

def load_from_file(fn):
    #Note: some areas that are not countries as: "Italy_Sacodc" do not contain
    #any of the words in selection but are still removed because they're if
    #they're in the InAreaName they'll always be paired with an area containing
    #one of the selection in the OutAreaName and vice versa.
    df = pd.read_csv('data/TP/'+fn, encoding = 'UTF-16', sep="\t",\
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
cbf.to_csv('processed/cbf_loaded.csv')

