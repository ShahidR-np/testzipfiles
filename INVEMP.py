import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import zipfile
import io
from xgboost import XGBClassifier
from urllib.error import URLError

st.set_page_config(page_title='INVEMP Tasty Bytes', page_icon='🍖🍕🍜')

st.sidebar.title("INVEMP: Customer Sales Calculation - Shahid")
st.sidebar.markdown("This web app allows you to view the likelihood of a group of customers churning. With the prevention of these customers churning, we can get an estimated calculation of the future revenue by this group of customers")

@st.cache_data
def load_orderdata():
     od = pd.read_csv("./orderdatav2.csv")
     return od


st.title('💲Customer Sales Calculation💲')
st.markdown("This tab predicts whether or not the customers in a selected cluster is likely to churn. It also includes insights on the selected cluster, such as their total revenue by year as well as the number of orders made by this cluster for each menu type.")
st.markdown("At the bottom, there is a revenue calculation to estimate the revenue by this cluster in the following year if they do not churn. This calculation is based on the cluster's revenue generated in the previous years.")
st.markdown('________________________________________________')

def read_csv_from_zipped_github(url):
# Send a GET request to the GitHub URL
   response = requests.get(url)
# Check if the request was successful
   if response.status_code == 200:
       # Create a BytesIO object from the response content
       zip_file = io.BytesIO(response.content)

       # Extract the contents of the zip file
       with zipfile.ZipFile(zip_file, 'r') as zip_ref:
           # Assume there is only one CSV file in the zip archive (you can modify this if needed)
           csv_file_name = zip_ref.namelist()[0]
           with zip_ref.open(csv_file_name) as csv_file:
               # Read the CSV data into a Pandas DataFrame
               df = pd.read_csv(csv_file)

       return df
   else:
       st.error(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
       return None

# gu_acd  =  "https://github.com/ShahidR-np/testzipfiles/raw/main/allcustdata.zip"
# custdata = read_csv_from_zipped_github(gu_acd)
gu_od = "https://github.com/ShahidR-np/testzipfiles/raw/main/orderdatav2.zip"
orderdata = read_csv_from_zipped_github(gu_od)

col1_t2, col2_t2, col3_t2 = st.columns(3)

# First dropdown list - Spending Level
with col1_t2:
     spending_level_t2 = st.selectbox("Customer Spending", ("High-Spending", "Average-Spending", "Low-Spending"))

# Second dropdown list - Frequency Level
with col2_t2:
     frequency_level_t2 = st.selectbox("Customer Frequency", ("High Frequency", "Average Frequency", "Low Frequency"))

# Third dropdown list - Age Level
with col3_t2:
     history_level_t2 = st.selectbox("Customer's History", ("Long-Standing", "Regular", "New"))

region_options = st.multiselect(
    'Region',
    ['California', 'Colorado', 'Massachusetts', 'New York', 'Washington'])
menu_Type_options = st.multiselect(
    'Menu Type',
    ['BBQ','Chinese','Crepes','Ethiopian','Grilled Cheese', 'Gyros', 'Hot Dogs', 'Ice Cream','Indian','Mac & Cheese','Poutine','Ramen','Sandwiches','Tacos','Vegetarian'])

CaliVal = 0
ColoVal = 0 
MasVal = 0
NewVal = 0
WashVal = 0




insight_button = st.button("Get Insights")
if insight_button:
     st.write(list(region_options))
     st.write(Region_options[[1]])
     #Variables
     generatedsales = 0 #The generate revenue for the cluster
     increasesales = 0 #The increase of revenue
     increaseperc = 0 #The increase of percentage in sales
     
     #Cluster vals
     freq_dict= {'High Frequency':0, 'Average Frequency':2, 'Low Frequency':1}
     spend_dict= {'High-Spending':0, 'Average-Spending':2, 'Low-Spending':1}
     hist_dict= {"Long-Standing":2, "Regular":0, "New":1}
     
     freq_val = freq_dict[frequency_level_t2]
     spend_val = spend_dict[spending_level_t2]
     hist_val = hist_dict[history_level_t2]
     if hist_val == 0:
        od = pd.read_csv("./custdatav0.csv")
     elif hist_val == 1:
        od = pd.read_csv("./custdatav1.csv")
     elif hist_val == 2:
        od = pd.read_csv("./custdatav2.csv")
     #Filtering data based on clusters
     #v1filtered = custdatav1[(custdatav1['sale_cluster'] == spend_val) & (custdatav1['Customer_age_cluster'] == hist_val) & (custdatav1['frequency_cluster'] == freq_val )]
     #v2filtered = custdatav2[(custdatav2['sale_cluster'] == spend_val) & (custdatav2['Customer_age_cluster'] == hist_val) & (custdatav2['frequency_cluster'] == freq_val )]
     filteredod = orderdata[(orderdata['sale_cluster'] == spend_val) & (orderdata['Customer_age_cluster'] == hist_val) & (orderdata['frequency_cluster'] == freq_val )]
     odgb = filteredod.groupby(['YEAR_OF_ORDER'])['ORDER_AMOUNT'].sum()
     #filteredcd = pd.concat([v1filtered, v2filtered])
     filteredcd = od[(od['sale_cluster'] == spend_val) & (od['frequency_cluster'] == freq_val )]
     clustermode = filteredcd.mode()
     gbmt = filteredod.groupby(['MENU_TYPE'])['MENU_TYPE'].count()
     gbmt = gbmt.sort_values(ascending = False)
     
     st.header("Insights")
     st.write("Total Sales by Year")
     st.bar_chart(odgb)
     st.write("Number of orders by menu type")
     st.table(gbmt)

# Model and Prediction

button_return_value = st.button("Predict")
if button_return_value:
     with open('cdc_xgb.pkl', 'rb') as file:
          cdcxgb = pickle.load(file)

     clustermode['frequency_cluster'] = freq_val
     clustermode['Customer_age_cluster'] = hist_val
     clustermode['sale_cluster'] = spend_val
     
     
     predictedchurn=cdcxgb.predict(clustermode[['TOTAL_PRODUCTS_SOLD', 'ORDER_AMOUNT', 'TOTAL_ORDERS',
       'MIN_DAYS_BETWEEN_ORDERS', 'MAX_DAYS_BETWEEN_ORDERS',
       'frequency_cluster', 'Customer_age_cluster', 'sale_cluster',
       'CITY_Boston', 'CITY_Denver', 'CITY_New York City', 'CITY_San Mateo',
       'CITY_Seattle', 'REGION_California', 'REGION_Colorado',
       'REGION_Massachusetts', 'REGION_New York', 'REGION_Washington',
       'MENU_TYPE_BBQ', 'MENU_TYPE_Chinese', 'MENU_TYPE_Crepes',
       'MENU_TYPE_Ethiopian', 'MENU_TYPE_Grilled Cheese', 'MENU_TYPE_Gyros',
       'MENU_TYPE_Hot Dogs', 'MENU_TYPE_Ice Cream', 'MENU_TYPE_Indian',
       'MENU_TYPE_Mac & Cheese', 'MENU_TYPE_Poutine', 'MENU_TYPE_Ramen',
       'MENU_TYPE_Sandwiches', 'MENU_TYPE_Tacos', 'MENU_TYPE_Vegetarian']])
     
     
     #predictedchurn = 1
     churntext = ""
     if (predictedchurn == 1):
          churntext = "LESS"
     else: 
          churntext = "MORE"
     
     odgb2022 = filteredod[filteredod['YEAR_OF_ORDER'] == 2022]
     odgb2021 = filteredod[filteredod['YEAR_OF_ORDER'] == 2021]
     odgb2020 = filteredod[filteredod['YEAR_OF_ORDER'] == 2020]
     odgb2019 = filteredod[filteredod['YEAR_OF_ORDER'] == 2019]
     avemth2022 = odgb[2022] / odgb2022['MONTH_OF_ORDER'].nunique()
     avemth2021 = odgb[2021] / odgb2021['MONTH_OF_ORDER'].nunique()
     avemth2020 = odgb[2020] / odgb2020['MONTH_OF_ORDER'].nunique()
     avemth2019 = odgb[2019] / odgb2019['MONTH_OF_ORDER'].nunique()
     
     percinc2020 = ((avemth2020-avemth2019)/avemth2019) * 100
     percinc2021 = ((avemth2021-avemth2020)/avemth2020) * 100
     percinc2022 = ((avemth2022-avemth2021)/avemth2021) * 100
     
     roi2021 = ((percinc2021 - percinc2020)/percinc2020) * 100
     roi2022 = ((percinc2022 - percinc2021)/percinc2021) * 100
     percinc2023 = ((100 + ((roi2021+roi2022)/2)) /100) * percinc2022
     avemth2023 = avemth2022 * ((100 + percinc2023)/100)
     odgb2023 = avemth2023 * 12
     
     
     
     generatedsales = odgb[2022]
     increasesales = odgb[2022] - odgb[2021]
     increaseperc = increasesales / generatedsales * 1002
     
     st.header("Prediction")
     st.write ("This cluster of customers is " + churntext + " likely to churn as compared to other clusters")
     st.write("After the implementation of discount coupon vouchers to these group of customer,")
     st.write("- The group of customer is less likely to churn the following year")
     
     st.header("Revenue Calculation")
     st.write("- In the year 2022, this group of customers have generated an average monthly sales of {0:.2f}".format(avemth2022))
     st.write("- In the following year, this group of customer is estimated to have an increase of {0:.2f}% in sales".format(percinc2023))
     st.write("- This translates into a estimated monthly sales of {0:.2f}, which is a total sales of {1:.2f}.".format(avemth2023, odgb2023))










