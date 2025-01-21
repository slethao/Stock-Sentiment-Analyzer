import pandas as pd
from sklearn.ensemble import IsolationForest

"""
—> include hashing… yeah…
handle missing values (imputation) **use the ARIMA (Autoregressive Integrated Moving Average) under the Time Series-Based Imputation
** (do this next), 
remove outliers, 
and address any inconsistencies.
"""


"""@TODO: fixed the group name to be the date"""
with open('Programmed/NVIDIA_STOCK.csv') as content:
    groups = content.readline().rstrip("\n")
    extra_line = content.readline().rstrip("\n")
    date_line = content.readline().rstrip("\n")
    date_group = date_line.split(",")[0]
    new_groups = groups.replace("Price", date_group)

    with open('Programmed/NVIDIA_STOCK_02.csv', 'w') as new_file:
        new_file.write(f"{new_groups}\n")
        for line in content:
            new_file.write(line)


"""@TODO: missing values and hashing"""

"""@TODO: remove outliers"""