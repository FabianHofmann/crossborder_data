# -*- coding: utf-8 -*-
"""
Created on Wed May 29 21:05:18 2019

@author: Hazem
"""
import pandas as pd
import auxiliary_functions as aux
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import matrix_rank as rank

# most nan are in 2016, if we drop this we have to filter out less
cbf_loaded = pd.read_csv('processed/cbf_loaded.csv', index_col=0,
                         parse_dates=True).loc['2017': '2018']

buses = np.unique(sum(cbf_loaded.columns.str.split(' - '), []))
total = [4,6,10,12,18,24,30,50]
consec = [2,3,4,8,10,12,18,25]

#this is a much faster approach
nans = aux.nan_statistics(cbf_loaded)

def K_from_index_list(idx):
    K = pd.DataFrame(index=np.unique(sum(idx.str.split(' - '), [])),
                     columns=idx)
    for i in idx:
        bus0, bus1 = i.split(' - ')
        K.at[bus0, i] = 1
        K.at[bus1, i] = -1
    return K.fillna(0)

def is_connected(K):
    return len(K) - rank(K) == 1

contained = pd.DataFrame(0, index=consec, columns=total)\
              .rename_axis(index='consec.', columns='total')\
              .stack()

for tot in total:
    for con in consec:
        included = nans\
            .query('max_total_per_month <= @tot and consecutive <= @con')\
            .index
        contained[con, tot] = included

border = 'ES - FR'
border_contained = contained.apply(lambda x: border in x).unstack()
number_contained = contained.apply(len).unstack()
connected = contained.apply(lambda idx: is_connected(K_from_index_list(idx)))\
                     .unstack()

print('remaining borders after filtering:\n', number_contained)
print(f'{border} contained after filtering:\n', border_contained)
print(f'network is connected:\n', connected)


stats = aux.get_monthly_aggregation().abs().sum()
#%%
df = contained.apply(lambda x: stats.reindex(x).sum()).sort_values()
contained = contained.reindex(df.index)
#df.index = [str(s).replace('(', '').replace(')', '') for s in df.index.values]

fig, ax = plt.subplots(figsize=(8,12))
df.loc['inf', 'inf'] = stats.sum()
df.div(1e3)[::-1].plot.barh(ax=ax)
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig('plots/filered_out_total_flow.png')

filtered_in_step = pd.Series(
        [contained.iloc[i].difference(contained.iloc[i-1]) if i else pd.Index([])
        for i in range(len(contained))],
        index=contained.index).apply(pd.Index.to_list)\
        .astype(str).str.replace('\[|\]','')\
        .to_frame('additional connector')\
        .to_html('plots/filered_out_connector_diff.html')

# does not work
#for ypos, val in filtered_in_step.iteritems():
#    ax.text(500, ypos, val)

#%%
nan_share = nans.query('total != 0').div(len(cbf_loaded))\
                .sort_values(by='total', ascending=False)

fig, ax = plt.subplots()
nan_share.rename(columns={'max_total_per_month': 'max total per month'})\
                .plot.bar(ax=ax)
fig.savefig(f'plots/nan_share_loaded.png')

