############################################
# CUSTOMER LIFETIME VALUE
############################################

# 1. Data Preparation
# 2. Average Order Value (average_order_value = total_price / total_transaction)
# 3. Purchase Frequency (total_transaction / total_number_of_customers)
# 4. Repeat Rate & Churn Rate (number of multiple customers / all customers)
# 5. Profit Margin (profit_margin =  total_price * 0.10)
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
# 8. Creating segments

##################################################
# 1. Data Preparation
##################################################

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x) # show 3 numbers after the comma

# Reading the dataset
df_ = pd.read_excel('datasets/csv_path/w3/online_retail_II.xlsx', sheet_name = 'Year 2010-2011') # 2010-2011 page is taken
df = df_.copy()
df.head()

# The 'C' in the invoice shows the canceled transactions. Canceled transactions leave the dataset.
df = df[~df["Invoice"].str.contains("C", na = False)]

# Data with quantity greater than zero will be evaluated
df = df[df["Quantity"] > 0]

# Remove the missing observations from the data set.
df.dropna(inplace = True)

# Earnings per invoice
df["TotalPrice"] = df["Quantity"] * df["Price"]

# Total transaction, total unit, total price values were calculated
cltv_c = df.groupby("Customer ID").agg({'Invoice': lambda x: x.nunique(),
                                'Quantity': lambda x: x.sum(),
                                'TotalPrice': lambda x: x.sum()})

cltv_c.columns = ['total_transaction','total_unit','total_price']
cltv_c.head()

##################################################
# 2. Average Order Value (average_order_value = total_price / total_transaction)
##################################################

cltv_c['avg_order_value'] = cltv_c['total_price'] / cltv_c['total_transaction']

##################################################
# 3. Purchase Frequency (total_transaction / total_number_of_customers)
##################################################

cltv_c['purchase_frequency'] =cltv_c['total_transaction'] / cltv_c.shape[0]

##################################################
# 4. Repeat Rate & Churn Rate (number of multiple customers / all customers)
##################################################

repeat_rate = cltv_c[cltv_c.total_transaction > 1].shape[0] / cltv_c.shape[0]
churn_rate = 1 - repeat_rate

##################################################
# 5. Profit Margin (profit_margin =  total_price * 0.10)
##################################################

cltv_c['profit_margin'] = cltv_c['total_price'] * 0.10
cltv_c.head()

##################################################
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
##################################################

cltv_c['customer_value'] = cltv_c['avg_order_value'] * cltv_c['purchase_frequency']

##################################################
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
##################################################

cltv_c['cltv'] = (cltv_c['customer_value'] / churn_rate) * cltv_c['profit_margin']

scaler = MinMaxScaler(feature_range = (0,1)) # converts values between 0-1
scaler.fit(cltv_c[["cltv"]]) # fit to cltv model
cltv_c["scaled_cltv"] = scaler.transform(cltv_c[['cltv']])

# sorts values from largest to smallest
cltv_c.sort_values(by = 'scaled_cltv', ascending = False).head()

##################################################
# 8. Creating segments
##################################################

# qcut = divides values by quartile
cltv_c["segment"] = pd.qcut(cltv_c["scaled_cltv"], 4, labels = ["D","C","B","A"])

cltv_c[["total_transaction","total_unit","total_price","cltv","scaled_cltv"]].agg({"count", "mean", "sum"})