import pandas as pd

instruments = ['AL','NI','CU','PB','AG',
               'RU','MA','PP','TA','L',
               'V','M','P','Y','OI',
               'C','CS','JD','SR','HC',
               'J','I','SF','RB','ZC',
               'FG']

df1 = pd.read_csv('D:\PythonProject\JD_FactorFactory01\DataJQ\Basis_xzh1.csv','r')
df1.fillna(0)
df1.index = df1['Unnamed: 0']
del df1['Unnamed: 0']
df1 = df1[instruments]