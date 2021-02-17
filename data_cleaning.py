# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

df = pd.read_csv('batters.csv')

#getting rid of last row
df=df[:-1]

#dropping positions played during season
df = df.drop('Pos Summary', axis = 1)

#don't want data from players traded during season
df = df[df['Lg'] != 'MLB']

# changing the 'Age' column for string to int
pd.to_numeric(df['Age'], errors = 'raise')
np.array_equal(df.Age, df.Age.astype(int))

#PLayer over age 30 or no
over_30=df[df['Age']>=30].index
df.loc[over_30,'over_30']=1
over_30=df[df['Age']<30].index
df.loc[over_30,'over_30']=0

#cleaning names
df['players'] = df['Name'].apply(lambda x: x.split('\\')[0].split('*')[0].split('#')[0])


#Plate appearances over 200
df = df[df['PA'] > 200]


df.columns

#rearrange columns
df = df[ ['players'] + [ col for col in df.columns if col != 'players' ] ]

df_out = df.drop(['Rk','Name'], axis = 1)

df_out.to_csv('mlb_data_cleaning.csv', index = False)


