# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 20:41:52 2018

@author: basil.p.sony
"""

# Required Python Packages
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#Importing Data
df = pd.read_excel('http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx')
df.head()

# Data Treatment - drop the rows that don’t have invoice numbers and remove the credit transactions
df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]

# One Hot coding
# 1) Consolidate the items into 1 transaction per row with each product for country France
basket = (df[df['Country'] =="France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

#  2) Converting positive values to a 1 and anything less than 0 is set to 0
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
basket_sets = basket.applymap(encode_units)

# Drop the column Postage
basket_sets.drop('POSTAGE', inplace=True, axis=1)

# Generating frequent item sets that have a support of at least 7% 
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)

# Generating the rules with their corresponding support, confidence and lift
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Saving the results generated
rules.to_csv('output.csv')

