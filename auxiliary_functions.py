# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 19:35:04 2019

@author: Hazem
"""
import pandas as pd
import matplotlib.pyplot as plt
import pycountry
from cartopy.io import shapereader
import numpy as np
import geopandas as gpd
cget = pycountry.countries.lookup
import pypsa

#%%Functions
def directed_cbf(df):
    oneway_b = ~df.columns.map(sorted).str.join(' - ').duplicated()

    positive_cbf = df.loc[:, oneway_b]
    negative_cbf = df.loc[:, ~oneway_b]\
                         .swaplevel(axis=1)\
                         .rename_axis(columns=df.columns.names)\
                         .sort_index(axis=1)

    one_na_one_zero = (positive_cbf.isnull() & (negative_cbf == 0)) | \
                      (negative_cbf.isnull() & (positive_cbf== 0))

    cbf = positive_cbf.sub(negative_cbf, fill_value=0)\
                      .where(~one_na_one_zero)

    cbf.columns = [" - ".join(v) for v in cbf.columns.values]
    return cbf

def directed_to_undirected(df):
    '''
    Extend cbf dataframe into unidirected
    '''
    positive, negative = df.clip(lower=0), df.clip(upper=0)
    negative = - negative.rename(columns=lambda s: s[-2:] + ' - ' + s[:2])
    return pd.concat([positive, negative], axis=1).sort_index(axis=1)


#def filter_data(df, total_nans, consecutive_nans):
#    """filters a given dataframe according to passed paramaters
#
#    Paramaters:
#    -----------
#    df (dataframe): ENTSOE physical cross-border flow data
#    total_nans (int): maximum number of nans allowed in a month for a connecter
#    consecutive_nans (int): maximum number of consecutive nans allowed in a month for a connector
#
#    Returns:
#    --------
#    df_filtered (dataframe): a filtered (and interpolated) dataframe.
#    """
#
#    nans = nan_statistics(df)
#    return df.reindex(columns=nans.query(
#                'max_total_per_month <= @total_nans and '
#                'consecutive <= @consecutive_nans').index)


def filter_data(df, alpha, threshold, per_year=False):
    if per_year:
        res = sum(gap_occurances(df, per_year=per_year))
    else:
        res = gap_occurances(df)
    res = res.apply(lambda x : x * x.index**alpha).sum().sort_values()
    return df.drop(res[res>threshold].index, axis=1)


def scale_data(df):
    stats = get_monthly_aggregation()*1000
    factors = df.resample('M').sum().to_period('M')\
                .pipe(lambda df: stats.reindex_like(df).div(df))\
                .resample('H', kind='timestamp').ffill()\
                .reindex(df.index).ffill()
    return df.multiply(factors)

def nan_statistics(df):
    def max_consecutive_nans(ds):
        return (ds.isnull().astype(int)
                  .groupby(ds.notnull().astype(int).cumsum())
                  .sum().max())
    consecutive = df.apply(max_consecutive_nans)
    total = df.isnull().sum()
    max_total_per_month = df.isnull().resample('m').sum().max()
    return pd.concat([total, consecutive, max_total_per_month],
                 keys=['total', 'consecutive', 'max_total_per_month'], axis=1)


def interpolate_data(df, how):
    return df.interpolate(how)


def gap_occurances(df, per_year=False, write=True):
    def consecutive_nans(ds):
        return (ds.isnull().astype(int)
                  .groupby(ds.notnull().astype(int).cumsum()).sum())
    if not per_year:
        return df.apply(lambda d: consecutive_nans(d).value_counts()).fillna(0)
    tables=[]
    for year, dff in df.reset_index().groupby(df.reset_index().DateTime.dt.year):

        cons_nans = dff.apply(consecutive_nans)
        table = pd.DataFrame(index=range(len(df)))

        for column in dff:
            table[column] = cons_nans[column].value_counts()

        table=table.fillna(0).drop(0).drop('DateTime', axis=1)
        if write:
            table.to_csv('processed/gap_occurances_'+str(year)+'.csv')
        tables.append(table)
    return tables


def average_week(df):
    groups = [df.index.dayofweek, df.index.hour]
    return df.groupby(groups).mean().rename_axis(index=['Day','Hour'])


def transform_to_average_week(df):
    groups = [df.index.dayofweek, df.index.hour]
    return df.groupby(groups).transform('mean')


#Export Gap Report
def Find_Gaps(df):
    """Function finds the gaps and write a simple report.

    Paramaters:
    -----------
    df(dataframe):

    """
    file = open("CrossBorderPhysicalFlow_Gaps.txt", "w")
    file.write('This file mentions only columns with total length of gaps > 1% of data\n\n')
    cols = []
    print('Amount of Missing Values:')
    for column in df:
        print(column+":", len(df[column][df[column].isnull()]))
        if df[column].isnull().values.any():
            if len(df[column][df[column].isnull()]) > 0.01 * len(df):
                cols.append(column)
                file.write('____________'+column+'___________\n')
                file.write('Number of Missing Values: '+ str(len(df[column][df[column].isnull()]))+'\n')
                file.write('First Missing Value: '+str(df[column][df[column].isnull()].index[0])+'\n')
                file.write('Last Missing Value: '+str(df[column][df[column].isnull()].index[-1])+'\n')
                file.write('\n\n\n')
    file.close()

def get_monthly_aggregation():
    exclude = ['AD', 'AM', 'AZ', 'MA', 'GE', 'IQ', 'IR', 'SY']
    return pd.read_csv('data/PS/PEPF.csv', skiprows=2)\
           .dropna()\
           .query('Submitted_By not in @exclude and Border_with not in @exclude')\
           .assign(
               Day=1,
               DateTime=lambda df: pd.to_datetime(df[['Year', 'Month', 'Day']],
                            format = '%Y-%M').dt.to_period('M'),
               AVG_Value =lambda df: pd.to_numeric(df.AVG_Value.str.replace(',',''),
                                                   errors='coerce'))\
           .drop(['Year', 'Month', 'Day'], axis=1)\
           .set_index('DateTime', drop=True)\
           .rename(columns={'Submitted_By': 'OutMapCode', 'Border_with': 'InMapCode',
                            'AVG_Value':'FlowValue'})\
           .pivot_table(index='DateTime', columns=['OutMapCode','InMapCode'],
                        values='FlowValue')\
           .pipe(directed_cbf)


def get_yearly_aggregation(value='Net'):
    """
    Loading function reading 'data/ENTSOE_imp-exp_bycountyry.csv' and formating
    data. Argument value can be one of 'Import', 'Export' or 'Net' (default).
    """
    return pd.read_csv('data/ENTSOE_imp-exp_bycountyry.csv')\
            .assign(DateTime=pd.Timestamp('2017'))\
            .pivot_table(index='DateTime', columns='Country', values=value)


def get_shapefiles(countries, resolution='50m', iso3=False):
    #set regions
    if not iso3:
        countries_iso3 = [cget(c).alpha_3 for c in countries]
    shpfilename = shapereader.natural_earth(
            resolution=resolution, category='cultural', name='admin_0_countries')
    reader = shapereader.Reader(shpfilename)
    records = reader.records()
    regions = pd.Series(
                {r.attributes['ADM0_A3']: r.geometry
                     if r.attributes['ADM0_A3'] in countries_iso3 else np.nan
                     for r in records})\
                .rename(index=dict(zip(countries_iso3, countries)))\
                .reindex(countries)
    return gpd.GeoDataFrame({'geometry': regions})

def create_network_from_cbf(cbfs):
    borders = pd.DataFrame(cbfs.columns.str.split(' - ').tolist(),
                       index=cbfs.columns, columns=['country_from', 'country_to'])
    buses = pd.DataFrame(index=borders.stack().unique())

    regions = get_shapefiles(buses.index)

    coords = pd.read_csv('processed/country_coords_revised.csv', index_col=1)
    buses = buses.assign(x = coords.x, y = coords.y)

    #setup network
    n = pypsa.Network()
    n.set_snapshots(cbfs.index)
    n.madd('Bus', names=buses.index, x=buses.x, y=buses.y)
    n.madd('Link', names=borders.index,
           bus0=borders.country_from, bus1=borders.country_to,
           p0 = cbfs, p1 = -cbfs)

    K = pd.DataFrame(pypsa.graph.incidence_matrix(n).todense(), index=buses.index,
                     columns=borders.index)

    n.buses_t.p = (K @ n.links_t.p0.T.fillna(0)).T
    n.regions = regions
    return n