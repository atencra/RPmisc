# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 17:28:53 2015

@author: craig
"""


import urllib

tb_deaths_url_csv = 'https://docs.google.com/spreadsheets/d/12uWVH_IlmzJX_75bJ3IH5E-Gqx6-zfbDKNvZqYjUuso/pub?gid=0&output=CSV'
tb_existing_url_csv = 'https://docs.google.com/spreadsheets/d/1X5Jp7Q8pTs3KLJ5JBWKhncVACGsg5v4xu6badNs4C7I/pub?gid=0&output=csv'
tb_new_url_csv = 'https://docs.google.com/spreadsheets/d/1Pl51PcEGlO9Hp4Uh0x2_QM0xVb53p2UDBMPwcnSjFTk/pub?gid=0&output=csv'

local_tb_deaths_file = 'tb_deaths_100.csv'
local_tb_existing_file = 'tb_existing_100.csv'
local_tb_new_file = 'tb_new_100.csv'

deaths_f = urllib.urlretrieve(tb_deaths_url_csv, local_tb_deaths_file)
existing_f = urllib.urlretrieve(tb_existing_url_csv, local_tb_existing_file)
new_f = urllib.urlretrieve(tb_new_url_csv, local_tb_new_file)

import pandas as pd

deaths_df = pd.read_csv(local_tb_deaths_file,index_col=0,thousands=',').T

existing_df = pd.read_csv(local_tb_existing_file,index_col=0,thousands=',').T

new_df = pd.read_csv(local_tb_new_file,index_col=0,thousands=',').T


existing_df.head()

existing_df.columns

existing_df.index



deaths_df.index.names = ['year']

deaths_df.columns.names = ['country']

existing_df.index.names = ['year']

existing_df.columns.names = ['country']

new_df.index.name = ['year']

new_df.columns.names = ['country']

existing_df


# Data indexing

existing_df['United Kingdom']

existing_df.Spain


existing_df[['Spain','United Kingdom']]


existing_df.Spain['1990']


existing_df[0:5]




# Indexing in production code
existing_df.iloc[0:2]

existing_df.loc['1992':'2005']


existing_df[['Spain','United Kingdom']].loc[['1992','1998','2005']]









