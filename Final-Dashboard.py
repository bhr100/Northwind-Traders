# Manipulation Packages
import numpy as np
import pandas as pd
import re
# Visualization Packages
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Dash Packages
from dash import Dash , html ,dcc
from dash.dependencies import Input ,Output
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import dash_mantine_components as dmc  

# Read Data
df = pd.read_csv('H:/CS50/Portfolio Projects/Northwind Traders/Data/Northwind.csv',
                 parse_dates=['orderDate','requiredDate','shippedDate'],
                 dtype={col :'category' for col in ['orderID', 'customerID', 'employeeID','productID','companyName_customers','contactName','productName','shipperID',
                                                    'contactTitle', 'city_customer', 'country_customer','categoryID', 'categoryName','employeeName','title', 
                                                    'city_employees', 'country_employees','companyName_shippers', 'Mon_Day_orderDate', 'Year_Quarter_orderDate',
                                                    'weeks_of_year_orderDate', 'Year_Month_orderDate','day_order','month_order', 'quarter_order', 'year_order', 'weekofyear_order']}
                 ).drop('Unnamed: 0',axis=1)


def get_data_matrices(*groubed_columns):
        data = df.groupby(list(groubed_columns))\
                            .agg(
                                    Total_Sales = ('Sales','sum'),
                                    Total_Orders = ('orderID','nunique'),
                                    Total_quantity = ('quantity','sum'),
                                    Average_Sales = ('Sales','mean'),
                                    Total_Transaction = ('orderID','count'), 
                                    Average_Orders = ('orderID',lambda x : x.nunique() / len(x)),
                                    life_value_day = ('orderDate', lambda x : (df.orderDate.max() - x.min()).days ),
                                    life_from_last_order = ('orderDate',lambda x : (df.orderDate.max() - x.max()).days),
                                    Total_Customer = ('customerID','nunique'),
                                    Total_Products = ('productID','nunique'),
                                    Total_discontinued_orders = ('discontinued',lambda x : x.astype('int').sum())
                                    )\
                                .assign(
                                    Average_Sales_Per_Order = lambda x : x['Total_Sales'] / x['Total_Orders'],
                                    Average_Sales_Per_Customer = lambda x : x['Total_Sales'] / x['Total_Customer'],
                                    Average_Sales_Per_Product = lambda x : x['Total_Sales'] / x['Total_Products'],
                                    Average_Orders_Per_Customer = lambda x : x['Total_Orders'] / x['Total_Customer'],
                                    Average_Orders_Per_Product = lambda x : x['Total_Orders'] / x['Total_Products'],
                                    Average_Sales_Per_Day = lambda x :  x['Total_Sales'] / x['life_value_day'],
                                    Average_Order_Per_Day = lambda x :  x['Total_Orders'] / x['life_value_day'],
                                    Average_Customer_Per_Day = lambda x :  x['Total_Customer'] / x['life_value_day'],
                                    Average_Product_Per_Day = lambda x :  x['Total_Products'] / x['life_value_day'],
                                    Average_Sales_Per_Month = lambda x :  x['Total_Sales'] / (x['life_value_day'] / 30),
                                    Average_Order_Per_Month = lambda x :  x['Total_Orders'] /( x['life_value_day'] / 30),
                                    Average_Customer_Per_Month = lambda x :  x['Total_Customer'] / (x['life_value_day'] / 30),
                                    Average_Product_Per_Month = lambda x :  x['Total_Products'] / (x['life_value_day'] / 30),
                                    Average_Sales_Per_Week = lambda x :  x['Total_Sales'] / (x['life_value_day'] / 7),
                                    Average_Order_Per_Week = lambda x :  x['Total_Orders'] /( x['life_value_day'] / 7),
                                    Average_Customer_Per_Week = lambda x :  x['Total_Customer'] / (x['life_value_day'] / 7),
                                    Average_Product_Per_Week = lambda x :  x['Total_Products'] / (x['life_value_day'] / 7),
                                    Average_Sales_Per_Quarter = lambda x :  x['Total_Sales'] / (x['life_value_day'] / 90),
                                    Average_Order_Per_Quarter = lambda x :  x['Total_Orders'] /( x['life_value_day'] / 90),
                                    Average_Customer_Per_Quarter = lambda x :  x['Total_Customer'] / (x['life_value_day'] / 90),
                                    Average_Product_Per_Quarter = lambda x :  x['Total_Products'] / (x['life_value_day'] / 90),
                                    Growth_Sales =  lambda x : (x['Total_Sales'].diff() /  x['Total_Sales']) * 100,
                                    Growth_Sales_Cumulative = lambda x : x['Growth_Sales'].rolling(window=len(x) , min_periods=1).sum(),
                                    Growth_Orders =  lambda x :( x['Total_Orders'].diff() /  x['Total_Orders']) * 100,
                                    Growth_Orders_Cumulative = lambda x : x['Growth_Orders'].rolling(window=len(x) , min_periods=1).sum(),
                                    Growth_Customer =  lambda x : (x['Total_Customer'].diff() /  x['Total_Customer']) * 100,
                                    Growth_Customer_Cumulative = lambda x : x['Growth_Customer'].rolling(window=len(x) , min_periods=1).sum(),
                                    Growth_Products =  lambda x : (x['Total_Products'].diff() /  x['Total_Products']) * 100,
                                    Growth_Products_Cumulative = lambda x : x['Growth_Products'].rolling(window=len(x) , min_periods=1).sum(),
                                    Pct_Sales_Cumulative = lambda x : round((x['Total_Sales'] / x['Total_Sales'].sum())*100,2).rolling(window=len(x) , min_periods=1).sum(),
                                    Pct_Orders_Cumulative = lambda x : round((x['Total_Orders'] / x['Total_Orders'].sum())*100,2).rolling(window=len(x) , min_periods=1).sum(),
                                    Pct_Customers_Cumulative = lambda x : round((x['Total_Customer'] / x['Total_Customer'].sum()*100),2).rolling(window=len(x) , min_periods=1).sum(),
                                    Pct_Products_Cumulative = lambda x : round((x['Total_Products'] / x['Total_Products'].sum()*100),2).rolling(window=len(x) , min_periods=1).sum(),
                                    Pct_diff_Sales_Cumulative = lambda x : x['Pct_Sales_Cumulative'].diff(),
                                    Pct_diff_Orders_Cumulative = lambda x : x['Pct_Orders_Cumulative'].diff(),
                                    Pct_diff_Customers_Cumulative = lambda x : x['Pct_Customers_Cumulative'].diff(),
                                    Pct_diff_Products_Cumulative = lambda x : x['Pct_Products_Cumulative'].diff(),
                                    Rolling_Average_Sales_Over_Month = lambda x : x['Total_Sales'].rolling(30).mean(),
                                    Rolling_Average_Orders_Over_Month = lambda x : x['Total_Orders'].rolling(30).mean(),
                                    Rolling_Average_Products_Over_Month = lambda x : x['Total_Products'].rolling(30).mean(),
                                    Rolling_Average_Customers_Over_Month = lambda x : x['Total_Customer'].rolling(30).mean(),
                                    Rolling_Average_Sales_Over_Week = lambda x : x['Total_Sales'].rolling(7).mean(),
                                    Rolling_Average_Orders_Over_Week = lambda x : x['Total_Orders'].rolling(7).mean(),
                                    Rolling_Average_Products_Over_Week = lambda x : x['Total_Products'].rolling(7).mean(),
                                    Rolling_Average_Customers_Over_Week = lambda x : x['Total_Customer'].rolling(7).mean(),
                                    Rolling_Average_Sales_Over_Quarter = lambda x : x['Total_Sales'].rolling(90).mean(),
                                    Rolling_Average_Orders_Over_Quarter = lambda x : x['Total_Orders'].rolling(90).mean(),
                                    Rolling_Average_Products_Over_Quarter = lambda x : x['Total_Products'].rolling(90).mean(),
                                    Rolling_Average_Customers_Over_Quarter = lambda x : x['Total_Customer'].rolling(90).mean(),
                                    Prior_Month_Sales = lambda x : x['Total_Sales'].shift(30),
                                    Prior_Month_Orders = lambda x : x['Total_Orders'].shift(30),
                                    Prior_Month_Products = lambda x : x['Total_Products'].shift(30),
                                    Prior_Month_Customers = lambda x : x['Total_Customer'].shift(30),
                                    Prior_Week_Sales = lambda x : x['Total_Sales'].shift(7),
                                    Prior_Week_Orders  = lambda x : x['Total_Orders'].shift(7),
                                    Prior_Week_Products = lambda x : x['Total_Products'].shift(7),
                                    Prior_Week_Customers  = lambda x : x['Total_Customer'].shift(7),
                                    Prior_Quarter_Sales = lambda x : x['Total_Sales'].shift(90),
                                    Prior_Quarter_Orders = lambda x : x['Total_Orders'].shift(90),
                                    Prior_Quarter_Products = lambda x : x['Total_Products'].shift(90),
                                    Prior_Quarter_Customer = lambda x : x['Total_Customer'].shift(90),
                                    Diff_Prior_Current_Month_Sales = lambda x : x['Prior_Month_Sales'] - x['Total_Sales'],
                                    Diff_Prior_Current_Month_Orders = lambda x : x['Prior_Quarter_Orders'] - x['Total_Orders'],
                                    Diff_Prior_Current_Month_Products = lambda x : x['Prior_Quarter_Products'] - x['Total_Products'],
                                    Diff_Prior_Current_Month_Customers = lambda x : x['Prior_Quarter_Customer'] - x['Total_Customer'],
                                    Diff_Prior_Current_Week_Sales = lambda x : x['Prior_Week_Sales'] - x['Total_Sales'],
                                    Diff_Prior_Current_Week_Orders = lambda x : x['Prior_Week_Orders'] - x['Total_Orders'],
                                    Diff_Prior_Current_Week_Products = lambda x : x['Prior_Week_Products'] - x['Total_Products'],
                                    Diff_Prior_Current_Week_Customers = lambda x : x['Prior_Week_Customers'] - x['Total_Customer'],
                                    Diff_Prior_Current_Quarter_Sales = lambda x : x['Prior_Quarter_Sales'] - x['Total_Sales'],
                                    Diff_Prior_Current_Quarter_Orders = lambda x : x['Prior_Quarter_Orders'] - x['Total_Orders'],
                                    Diff_Prior_Current_Quarter_Products = lambda x : x['Prior_Quarter_Products'] - x['Total_Products'],
                                    Diff_Prior_Current_Quarter_Customers = lambda x : x['Prior_Quarter_Customer'] - x['Total_Customer'],


                                )\
                            .reset_index() 
        return data
Comparison_Columns = ['Growth_Sales',
                    'Growth_Sales_Cumulative', 'Growth_Orders', 'Growth_Orders_Cumulative',
                    'Growth_Customer', 'Growth_Customer_Cumulative', 'Growth_Products',
                    'Growth_Products_Cumulative', 'Pct_Sales_Cumulative',
                    'Pct_Orders_Cumulative', 'Pct_Customers_Cumulative',
                    'Pct_Products_Cumulative', 'Pct_diff_Sales_Cumulative',
                    'Pct_diff_Orders_Cumulative', 'Pct_diff_Customers_Cumulative',
                    'Pct_diff_Products_Cumulative', 'Rolling_Average_Sales_Over_Month',
                    'Rolling_Average_Orders_Over_Month',
                    'Rolling_Average_Products_Over_Month',
                    'Rolling_Average_Customers_Over_Month',
                    'Rolling_Average_Sales_Over_Week', 'Rolling_Average_Orders_Over_Week',
                    'Rolling_Average_Products_Over_Week',
                    'Rolling_Average_Customers_Over_Week',
                    'Rolling_Average_Sales_Over_Quarter',
                    'Rolling_Average_Orders_Over_Quarter',
                    'Rolling_Average_Products_Over_Quarter',
                    'Rolling_Average_Customers_Over_Quarter', 'Prior_Month_Sales',
                    'Prior_Month_Orders', 'Prior_Month_Products', 'Prior_Month_Customers',
                    'Prior_Week_Sales', 'Prior_Week_Orders', 'Prior_Week_Products',
                    'Prior_Week_Customers', 'Prior_Quarter_Sales', 'Prior_Quarter_Orders',
                    'Prior_Quarter_Products', 'Prior_Quarter_Customer',
                    'Diff_Prior_Current_Month_Sales', 'Diff_Prior_Current_Month_Orders',
                    'Diff_Prior_Current_Month_Products',
                    'Diff_Prior_Current_Month_Customers', 'Diff_Prior_Current_Week_Sales',
                    'Diff_Prior_Current_Week_Orders', 'Diff_Prior_Current_Week_Products',
                    'Diff_Prior_Current_Week_Customers', 'Diff_Prior_Current_Quarter_Sales',
                    'Diff_Prior_Current_Quarter_Orders',
                    'Diff_Prior_Current_Quarter_Products',
                    'Diff_Prior_Current_Quarter_Customers']
# Create Dash App
with open('H:/CS50/Portfolio Projects/Northwind Traders/assets/reset.css','w') as file:
        file.write('body{background-color:rgb(34, 34, 34);}\n')
        file.close()

dbc_css = "/assets/reset.css"

app = Dash(__name__,external_stylesheets=[dbc.themes.DARKLY,dbc_css] )


load_figure_template('DARKLY')

style = {
    'borderTop': '1px solid #d6d6d6',
    'border-left': '1px solid #d6d6d6',
    'border-right': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': 'rgb(30, 30, 30)',
    'color': 'white',
    'font-family':'Arile',
    'font-size':15,
     
}  
tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'border-left': '1px solid #d6d6d6',
    'border-right': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#666666',
    'color': 'white',
    'font-family':'Arile',
    'font-size':15,
    'padding': '5px' 
} 
styles_cheps = {
    "label": {
        "&[data-checked]": {
            "&, &:hover": {
                "backgroundColor": dmc.theme.DEFAULT_COLORS["dark"][5],
                "color": "white",
            },
        },
    }
}


app.layout = dbc.Container(
                           fluid=True,
                           style={'backgroundColor':'rgb(34, 34, 34)','color':'white','border':'rgb(34, 34, 34)'},
                           children=[
                                     dcc.Tabs([
                                               dcc.Tab(
                                                       label='Trends',
                                                       className='dbc',
                                                       style = style,
                                                       selected_style=tab_selected_style,
                                                       children=[
                                                                 html.Br(),
                                                                 dmc.Grid(
                                                                        children=[
                                                                                 dmc.Col([
                                                                                        dcc.Markdown('__Select One Of The Matrices__'),
                                                                                        dcc.Dropdown(
                                                                                            options=[{'label': html.Span(col.replace("_"," "), style={'color': 'black'}) ,'value':col } 
                                                                                                  for col in get_data_matrices('orderDate').columns[1:].to_list()],
                                                                                            value='Total_Sales',
                                                                                            id='Matrices',
                                                                                            className='dbc')
                                                                                            ],span=3),
                                                                                dmc.Col([
                                                                                        dcc.Markdown('__Select The Date Matrices__'),
                                                                                        dcc.Dropdown(
                                                                                            options=[{'label': html.Span(col.replace("_"," "), style={'color': 'black'}) if "_" in col else html.Span(col, style={'color': 'black'}), 'value': col} 
                                                                                                     for col in ['orderDate','requiredDate','shippedDate','Mon_Day_orderDate',  
                                                                                                                 'Year_Quarter_orderDate','Mon_Day_orderDate','Year_Quarter_orderDate','day_order','month_order', 
                                                                                                                  'quarter_order', 'year_order', 'weekofyear_order']],
                                                                                            id='date_groupe',
                                                                                            value='orderDate',
                                                                                            className='dbc'),
                                                                                      ] ,span=3),
                                                                               dmc.Col([
                                                                                        dcc.Markdown('__Select The Comparison Matrices__'),
                                                                                        dcc.Dropdown(
                                                                                            options=[{'label': html.Span(col.replace("_"," "), style={'color': 'black'}) if "_" in col else html.Span(col, style={'color': 'black'}), 'value': col} for col in Comparison_Columns],
                                                                                            id='comparison',
                                                                                            value='Growth_Sales',
                                                                                            className='dbc'),
                                                                                      ] ,span=3),       
                                                                                dmc.Col([
                                                                                        dcc.Markdown('__Select a Range Of Date__'),
                                                                                            dmc.DateRangePicker(
                                                                                                id='rang_Date',
                                                                                                className='dbc'),
                                                                                      ] ,span=3),            
                                                                                dmc.Col([
                                                                                        dcc.Graph(id='Sales-Card',className='dbc') ,
                                                                                        ],span=2),
                                                                                dmc.Col([
                                                                                        dcc.Graph(id='Card-1',className='dbc') ,
                                                                                        ],span=2),
                                                                                dmc.Col([
                                                                                        dcc.Graph(id='Card-2',className='dbc') ,
                                                                                        ],span=2),
                                                                                dmc.Col([
                                                                                        dcc.Graph(id='Card-3',className='dbc') ,
                                                                                        ],span=2) ,
                                                                                dmc.Col([
                                                                                        dcc.Graph(id='Card-4',className='dbc') ,
                                                                                        ],span=2) ,
                                                                                dmc.Col([
                                                                                        dcc.Graph(id='Card-5',className='dbc') ,
                                                                                        ],span=2)                                          
                                                                                ],justify="center",align="stretch"),
                                                                 dmc.Grid(
                                                                        children=[
                                                                                dmc.Col(dmc.Stack(
                                                                                            [
                                                                                            dcc.Graph(id='Card_6',style={"height": 200,'width':280}),
                                                                                            dcc.Graph(id='Card_7',style={"height": 200,'width':280}),
                                                                                            dcc.Graph(id='Card_8',style={"height": 200,'width':280}),
                                                                                            ],
                                                                                            style={"height": 4500},
                                                                                            align="stretch",
                                                                                            justify="flex-start",
                                                                                            spacing='xs',
                                                                                        ),span=1),
                                                                                dmc.Col(dmc.Stack(
                                                                                            [
                                                                                            dcc.Graph(id='Card_9',style={"height": 200,'width':280}),
                                                                                            dcc.Graph(id='Card_10',style={"height": 200,'width':280}),
                                                                                            dcc.Graph(id='Card_11',style={"height": 200,'width':280}),
                                                                                            ],
                                                                                            style={"height": 4500},
                                                                                            align="stretch",
                                                                                            justify="flex-start",
                                                                                            spacing='xs',
                                                                                        ),span=1),      
                                                                                dmc.Col(dcc.Graph(id='Line_Chart',style={"height": 620}),span=9)
                                                                                 ],justify="space-between")         

                                                                 ],
                                                      ),
                                                      dcc.Tab(
                                                              label='Best & Worest',
                                                              className='dbc',
                                                              style=style,
                                                              selected_style=tab_selected_style,
                                                              children=[
                                                                        html.Br(),
                                                                        dmc.Grid([
                                                                                  dmc.Col([
                                                                                           
                                                                                           dmc.Group(children=[
                                                                                                     dmc.Switch(
                                                                                                            size='xl',
                                                                                                            radius='lg',
                                                                                                            onLabel="Less",
                                                                                                            offLabel="Top",
                                                                                                            checked=False,
                                                                                                            color='gray',
                                                                                                            id='switch',
                                                                                                            className='dbc'
                                                                                                              ),
                                                                                                     dcc.Dropdown(
                                                                                                               options=[{'label':html.Span(col.replace("_"," "), style={'color': 'black'})  , 'value':col} for col in ['companyName_customers','contactName','contactTitle','city_customer','country_customer',
                                                                                                                                                                             'productName','categoryName','employeeName','title','city_employees', 'country_employees']],
                                                                                                                id='Dropdown',
                                                                                                                optionHeight=50,
                                                                                                                # value='contactName',
                                                                                                                placeholder="Select a column",
                                                                                                                style={'width':150},
                                                                                                                className='dbc'
                                                                                                              ),         
                                                                                                     dmc.SegmentedControl(
                                                                                                                          data=[{'label':col.replace("_"," ") , 'value':col} for col in ['Total_Sales','Total_freight','Total_Orders' ,
                                                                                                                                                                                       'Total_Transaction' ,'Total_Products' ,'Total_Customers']],
                                                                                                                           id="segmented",
                                                                                                                           style={'width':620},
                                                                                                                           value='Total_Sales',
                                                                                                                           color='gray',
                                                                                                                           className='dbc')
                                                                                                               ],
                                                                                                    ),
                                                                                           html.Br(),  
                                                                                           dcc.Graph(id='Bar_Chart')
                                                                                          ],span=6),
                                                                                          dmc.Col([
                                                                                               dmc.SimpleGrid(
                                                                                                   cols=2,
                                                                                                   spacing='xl',
                                                                                                   children =[
                                                                                                        dcc.Dropdown(
                                                                                                               options=[{'label':html.Span(col.replace("_"," "), style={'color': 'black'})  , 'value':col} for col in ['companyName_customers','contactName','contactTitle','city_customer','country_customer',
                                                                                                                                                                             'productName','categoryName','employeeName','title','city_employees', 'country_employees']],
                                                                                                                id='Dropdown_X',
                                                                                                                optionHeight=50,
                                                                                                                placeholder="Select X Column",
                                                                                                                value = 'employeeName',
                                                                                                                className='dbc'
                                                                                                              ),
                                                                                                       dcc.Dropdown(
                                                                                                               options=[{'label':html.Span(col.replace("_"," "), style={'color': 'black'})  , 'value':col} for col in ['companyName_customers','contactName','contactTitle','city_customer','country_customer',
                                                                                                                                                                             'productName','categoryName','employeeName','title','city_employees', 'country_employees']],
                                                                                                                id='Dropdown_Y',
                                                                                                                optionHeight=50,
                                                                                                                placeholder="Select Y Column",
                                                                                                                value='title',
                                                                                                                className='dbc'
                                                                                                              )       
                                                                                                        
                                                                                                   ]),
                                                                                               html.Br(),    
                                                                                               dcc.Graph(id='heatmap')
                                                                                                   ]
                                                                                                  ,span=6)
                                                                                 ]),
                                                                               
                                                                        dmc.Grid([
                                                                                  html.Br(),  
                                                                                  dmc.Col([
                                                                                          dmc.SimpleGrid(
                                                                                                   cols=2,
                                                                                                   spacing='xl',
                                                                                                   children=[
                                                                                                    dcc.Dropdown(
                                                                                                                options=[{'label':html.Span(col.replace("_"," "), style={'color': 'black'})  , 'value':col} for col in ['Mon_Day_orderDate', 'Year_Quarter_orderDate','weeks_of_year_orderDate', 'Year_Month_orderDate',
                                                                                                                                                                                    'day_order','month_order', 'quarter_order', 'year_order', 'weekofyear_order']],
                                                                                                                            id='x_column_dropdown',
                                                                                                                            optionHeight=50,
                                                                                                                            value='day_order',
                                                                                                                            placeholder="Select x column",
                                                                                                                            className='dbc'
                                                                                                                    ),
                                                                                                    dmc.ChipGroup(
                                                                                                                    [dmc.Chip(x, value=x,variant='filled',color='gray',style=styles_cheps,size='md') for x in ["Reset","Top 10","Less 10"]],
                                                                                                                    id='chips',
                                                                                                                    value='Reset'
                                                                                                                    ), 
                                                                                                        ]),               
                                                                                          dcc.Graph(id='Pct_Area_Chart',hoverData={'points':[{'customdata':['']}]})  
                                                                                              
                                                                                         ],span=7),
                                                                                  html.Br(), 
                                                                                  dmc.Col([
                                                                                          dcc.Graph(id='Line_Hover_Area')
                                                                                         ],span=5)       

                                                                                 ])         

                                                                       ]
                                                             ),
                                                      dcc.Tab(
                                                              label='Ranks',
                                                              className='dbc',
                                                              style=style,
                                                              selected_style=tab_selected_style,
                                                              children=[
                                                                        html.Br(),
                                                                        html.H3('Analyze Ranks For Each Group By Columns',style={'text-align':'center'}),
                                                                        html.Hr(),
                                                                        dmc.SimpleGrid(
                                                                             cols=10,
                                                                             children=[
                                                                                  dcc.Graph(id='rank_1',style={"height": 150,'width':150}),
                                                                                  dcc.Graph(id='rank_2',style={"height": 150,'width':150}),
                                                                                  dcc.Graph(id='rank_3',style={"height": 150,'width':150}),
                                                                                  dcc.Graph(id='rank_4',style={"height": 150,'width':150}),
                                                                                  dcc.Graph(id='rank_5',style={"height": 150,'width':150}),
                                                                                  dcc.Graph(id='rank_6',style={"height": 150,'width':150}),
                                                                                  dcc.Graph(id='rank_7',style={"height": 150,'width':150}),
                                                                                  dcc.Graph(id='rank_8',style={"height": 150,'width':150}),
                                                                                  dcc.Graph(id='rank_9',style={"height": 150,'width':150}),
                                                                                  dcc.Graph(id='rank_10',style={"height": 150,'width':150}),
                                                                                     ],
                                                                              spacing='xs'),      
                                                                        dmc.SimpleGrid(
                                                                             cols=10,
                                                                             children=[
                                                                                  dcc.Graph(id='rank_11',style={"height": 150,'width':150}),
                                                                                  dcc.Graph(id='rank_12',style={"height": 150,'width':150}),
                                                                                  dcc.Graph(id='rank_13',style={"height": 150,'width':150}),
                                                                                  dcc.Graph(id='rank_14',style={"height": 150,'width':150}),
                                                                                  dcc.Graph(id='rank_15',style={"height": 150,'width':150}),
                                                                                  dcc.Graph(id='rank_16',style={"height": 150,'width':150}),
                                                                                  dcc.Graph(id='rank_17',style={"height": 150,'width':150}),
                                                                                  dcc.Graph(id='rank_18',style={"height": 150,'width':150}),
                                                                                  dcc.Graph(id='rank_19',style={"height": 150,'width':150}),
                                                                                  dcc.Graph(id='rank_20',style={"height": 150,'width':150}),
                                                                                     ],
                                                                            spacing='xs'),
                                                                        html.Br(),    
                                                                        dmc.Grid([
                                                                              dmc.Col([
                                                                                        dcc.Markdown('__Select Groupby Columns__'),
                                                                                        dcc.Dropdown(
                                                                                            options=[{'label': html.Span(col.replace("_"," ") if "_" in col else col ,style={'color':'black'} ), 'value': col} for col in ['companyName_customers', 'contactName','contactTitle', 'city_customer', 'country_customer','categoryName',
                                                                                                                                                                                        'productName','employeeName', 'title','city_employees', 'country_employees']],
                                                                                            id='groupby_columns',
                                                                                            value='contactName',
                                                                                            className='dbc'),
                                                                                      ] ,span=4),
                                                                              dmc.Col([
                                                                                        dcc.Markdown('__Select X Column__'),
                                                                                        dcc.Dropdown(
                                                                                            options=[{'label': html.Span(col.replace("_"," ") if "_" in col else col ,style={'color':'black'}) , 'value': col} for col in ['Total_Sales', 'Total_Orders', 'Total_quantity','Average_Sales', 'Total_Transaction','Average_Orders','life_value_day' ,
                                                                                                                                                                                         'life_from_last_order', 'Total_Customer','Total_Products', 'Total_discontinued_orders']],
                                                                                            id='x_column',
                                                                                            value='Total_Sales',
                                                                                            className='dbc'),
                                                                                      ] ,span=2),
                                                                               dmc.Col([
                                                                                        dcc.Markdown('__Select Y Column__'),
                                                                                        dcc.Dropdown(
                                                                                            options=[{'label': html.Span(col.replace("_"," ") if "_" in col else col ,style={'color':'black'}) , 'value': col} for col in ['Total_Sales', 'Total_Orders', 'Total_quantity','Average_Sales', 'Total_Transaction','Average_Orders','life_value_day' ,
                                                                                                                                                                                         'life_from_last_order', 'Total_Customer','Total_Products', 'Total_discontinued_orders']],
                                                                                            id='y_column',
                                                                                            value='Total_Orders',
                                                                                            className='dbc'),
                                                                                      ] ,span=2),
                                                                               dmc.Col([
                                                                                        dcc.Markdown('__Select Color Column__'),
                                                                                        dcc.Dropdown(
                                                                                            options=[{'label': html.Span(col.replace("_"," ") if "_" in col else col ,style={'color':'black'})  , 'value': col} for col in ['Average_Sales_Per_Order', 'Average_Sales_Per_Customer','Average_Sales_Per_Product', 'Average_Orders_Per_Customer','Average_Orders_Per_Product', 
                                                                                                                                                                                        'Average_Sales_Per_Day','Average_Order_Per_Day', 'Average_Customer_Per_Day','Average_Product_Per_Day', 'Average_Sales_Per_Month','Average_Order_Per_Month',
                                                                                                                                                                                          'Average_Customer_Per_Month','Average_Product_Per_Month', 'Average_Sales_Per_Week','Average_Order_Per_Week', 'Average_Customer_Per_Week','Average_Product_Per_Week',
                                                                                                                                                                                            'Average_Sales_Per_Quarter','Average_Order_Per_Quarter', 'Average_Customer_Per_Quarter','Average_Product_Per_Quarter']],
                                                                                            id='color_column',
                                                                                            value='Average_Sales_Per_Order',
                                                                                            className='dbc'),
                                                                                      ] ,span=2),
                                                                                dmc.Col([
                                                                                        dcc.Markdown('__Select Size Column__'),
                                                                                        dcc.Dropdown(
                                                                                            options=[{'label': html.Span(col.replace("_"," ") if "_" in col else col ,style={'color':'black'}), 'value': col} for col in ['Average_Sales_Per_Order', 'Average_Sales_Per_Customer','Average_Sales_Per_Product', 'Average_Orders_Per_Customer','Average_Orders_Per_Product', 
                                                                                                                                                                                        'Average_Sales_Per_Day','Average_Order_Per_Day', 'Average_Customer_Per_Day','Average_Product_Per_Day', 'Average_Sales_Per_Month','Average_Order_Per_Month',
                                                                                                                                                                                          'Average_Customer_Per_Month','Average_Product_Per_Month', 'Average_Sales_Per_Week','Average_Order_Per_Week', 'Average_Customer_Per_Week','Average_Product_Per_Week',
                                                                                                                                                                                            'Average_Sales_Per_Quarter','Average_Order_Per_Quarter', 'Average_Customer_Per_Quarter','Average_Product_Per_Quarter']],
                                                                                            id='size_column',
                                                                                            value='Average_Sales_Per_Quarter',
                                                                                            className='dbc'),
                                                                                      ] ,span=2),
                                                                                dmc.Col([
                                                                                         dcc.Graph(id='scatter_chart',className='dbc',hoverData={'points':[{'customdata':['Roland Mendel']}]})
                                                                                        ],span=12) ]                            
                                                                        )             
                                                                       ]
                                                              ),
                                                     dcc.Tab(
                                                            label='Scenario',
                                                            className='dbc',
                                                            style=style,
                                                            selected_style=tab_selected_style,
                                                            children=[
                                                                      html.Br(),
                                                                      html.H1('Scenarios Analysis',style={'text-align':'center'}),
                                                                      html.Hr(),
                                                                      dmc.Grid([
                                                                                dmc.Col([
                                                                                        dcc.Markdown('__Drag The Percentage Of Scenario__'),
                                                                                         dmc.Slider(
                                                                                                    showLabelOnHover=True,
                                                                                                    id='slider_scenario',
                                                                                                    updatemode="drag",
                                                                                                    marks=[{'label':f'%{round(i*100)}' , 'value': round(i,2)} for i in list(np.linspace(-0.5 , 0.5 , 5))],
                                                                                                    value=0,
                                                                                                    min=-.50,
                                                                                                    max=.50,
                                                                                                    step=0.05,
                                                                                                    radius='xs',
                                                                                                    size='xl',
                                                                                                    color='gray'
                                                                                         )
                                                                                        ],span=4),
                                                                                dmc.Col([
                                                                                        dcc.Markdown('__Select The Date Matrice__'),
                                                                                        dcc.Dropdown(
                                                                                                    options=[{'label':html.Span(col.replace("_"," "), style={'color': 'black'}),'value':col} for col in ['orderDate','Mon_Day_orderDate', 'Year_Quarter_orderDate','weeks_of_year_orderDate', 'Year_Month_orderDate',
                                                                                                                                                                                    'day_order','month_order', 'quarter_order', 'year_order', 'weekofyear_order']],
                                                                                                    id='Scenario_dropdown',
                                                                                                    value='orderDate',
                                                                                                    className='dbc'
                                                                                                                    ),
                                                                                         ],span=4),
                                                                                dmc.Col([
                                                                                        dcc.Markdown('__Select Any Of The Following Products__'),
                                                                                        dcc.Dropdown(
                                                                                                    options=[{'label':html.Span(col , style={'color':'black'}) , 'value':col}for col in df.productName.unique().tolist()],
                                                                                                    id='Scenario_Multi_Products',
                                                                                                    className='dbc',
                                                                                                    value = None ,
                                                                                                    multi=True,
                                                                                                    placeholder='Select Multi Products',
                                                                                                    style={'border':'white'},
                                                                                                                    ),
                                                                                         ],span=4),
                                                                                dmc.Col([
                                                                                    dmc.Stack(
                                                                                            [
                                                                                            dcc.Graph(id='Card_Senario_1',style={"height": 300,'width':300}),
                                                                                            dcc.Graph(id='Card_Senario_2',style={"height": 300,'width':300}),
                                                                                            ],
                                                                                            # style={"height": 8000},
                                                                                            align="stretch",
                                                                                            justify="center",
                                                                                            spacing='xl',
                                                                                        ),
                                                                                        ],span=2),
                                                                                dmc.Col([
                                                                                        dmc.Stack(
                                                                                            [
                                                                                            dcc.Graph(id='Card_Senario_3',style={"height": 300,'width':300}),
                                                                                            dcc.Graph(id='Card_Senario_4',style={"height": 300,'width':300}),
                                                                                            ],
                                                                                            # style={"height": 8000},
                                                                                            align="stretch",
                                                                                            justify="center",
                                                                                            spacing='xl',
                                                                                        )
                                                                                ],span=2) ,                         
                                                                                dmc.Col([
                                                                                         dcc.Graph(id='line_scenario')
                                                                                        ],span=8)        
                                                                              ])
                                                                      ]
                                                             
                                                     )

          
                                              ])
                                    ])
#TAP-ONE#############################################################################################################################
@app.callback(
          Output('rang_Date','value'),
          Output('rang_Date','minDate'),
          Output('rang_Date','maxDate'),
          Input('date_groupe','value')
)
def date_range(date_column):
        
        return [df[date_column].min() , df[date_column].max()] , df[date_column].min() , df[date_column].max()
        
@app.callback(
              Output('Sales-Card','figure'),
              Output('Card-1','figure'),
              Output('Card-2','figure'),
              Output('Card-3','figure'),
              Output('Card-4','figure'),
              Output('Card-5','figure'),
              Output('Card_6','figure'),
              Output('Card_7','figure'),
              Output('Card_8','figure'),
              Output('Card_9','figure'),
              Output('Card_10','figure'),
              Output('Card_11','figure'),
              Output('Line_Chart','figure'),
              Input('Matrices','value'),
              Input('date_groupe','value'),
              Input('rang_Date','value'),
              Input('comparison','value')

             )
def Cards(matrice , date , range , compare_line):
    
    card_data = get_data_matrices(date)
    try:
        if str(date) in ['orderDate','requiredDate','shippedDate']:
            df = round(get_data_matrices(date))
            card_data = card_data[card_data[date].between(range[0],range[1])]
            pct_range_slider = round(df[date].between(range[0],range[1]).mean() * 100 , 2)
            range_1 = round(card_data[df[date].between(range[0],range[1])][matrice].min() * 100 , 2)
            range_2 = round(card_data[df[date].between(range[0],range[1])][matrice].max() * 100 , 2)
            title = f'<b>{pct_range_slider}% of {matrice.replace("_"," ")} are between {range_1:.0f} to {range_2:.0f}</b>'
        else:
            card_data
            title = f'<b>{matrice.replace("_"," ")} By {date.replace("_"," ")}</b>'       
    except TypeError:
        card_data
        title = f'<b>{matrice.replace("_"," ")} By {date.replace("_"," ")}</b>'  
         
                 
          

    def card_fig(col):
        mask_reference_value = card_data.loc[(card_data[date].idxmax() - 2 ) if card_data.loc[card_data[date].idxmax(),col] == card_data.loc[card_data[date].idxmax()-1,col] else (card_data[date].idxmax() - 1), col]
        fig = go.Figure(go.Indicator(
            mode = "number+delta",
            value = card_data.loc[card_data[date].idxmax(),col],
            number = {"prefix": "$",'valueformat':",.2s",'font_size':40},
            delta = {"reference": mask_reference_value, "valueformat": ",.2s", "prefix": "$",'font_size':25},
            domain = {'y': [0.8, 1]}
            ))
        fig.add_scatter(x=card_data[date] , y=card_data[col],
        fill='tozeroy',
        marker_color='#d0d0e1',
        line=dict(width=0.5) ,hoverinfo='x+y' )
        fig.update_xaxes(visible=False, fixedrange=True)
        fig.update_yaxes(visible=False, fixedrange=True)
        fig.update_layout(xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=False),
                        plot_bgcolor='rgb(34, 34, 34)',
                        paper_bgcolor = 'rgb(34, 34, 34)' ,
                        height=300,
                        width=300,
                        title=f'{col.replace("_"," ")}',
                        title_x=0.5 ,
                        title_font_size=20,
                        font_family='Raleway')
        return fig
    
    def card_fig2(col):
        fig =   go.Figure(go.Indicator(
                    mode = "number",
                    value = card_data[col].sum(),
                    number = {'valueformat': ",.2s" if col == "Total_Sales" else ",.0f",'font_size':40 }
                    ))\
                .update_layout(height=200,width=200,title = f'{col.replace("_"," ")}',title_x = 0.5  , 
                        font_family='Raleway',plot_bgcolor='rgb(34, 34, 34)',paper_bgcolor = 'rgb(34, 34, 34)')
        return fig
    

    line = go.Figure()
    line.add_trace(go.Scatter(
                            x=card_data[date],
                            y=card_data[matrice],
                            line_shape='spline',
                            hoverinfo='x+y',
                            name=matrice.replace('_',' '),
                            hovertemplate="<br><b>%{x}</b><br>"+"<b>%%{y:,.2s}%</b>" if re.findall(r'(Pct)|(Growth)',matrice)else "<br><b>%{x}</b><br>"+"<b>$%{y:,.2s}</b>"  ,
                            marker_color='#5c5c8a'
                            ))
    line.add_trace(go.Scatter(
                            x=card_data[date],
                            y=card_data[compare_line],
                            line_shape='spline',
                            hoverinfo='x+y',
                            name=compare_line.replace('_',' '),
                            hovertemplate="<br><b>%{x}</b><br>"+"<b>%%{y:,.2s}</b>" if re.findall(r'(Pct)|(Growth)',compare_line) else "<br><b>%{x}</b><br>"+"<b>$%{y:,.2s}</b>"  ,
                            line=dict(dash='dot',color='#6600ff'),
                            ))                        
    line.update_layout(
                    title=title,
                    title_x=0.5,
                    font_family='Raleway',
                    title_font_size = 20,
                    plot_bgcolor ='rgb(34, 34, 34)',
                    paper_bgcolor = 'rgb(34, 34, 34)',
                    xaxis=dict(showgrid=False,title=''),
                    yaxis=dict(showgrid=False,title='',visible=False),
                    hovermode='x unified',
                    legend=dict(x=0.8,y=1.5,orientation='h'))\
            .update_xaxes(rangeslider_visible=True)                  

    
    return card_fig('Total_Sales'),card_fig('Average_Sales'),card_fig('Average_Sales_Per_Order'),card_fig('Average_Sales_Per_Customer'),card_fig('Average_Sales_Per_Order'),card_fig('Average_Sales_Per_Product'),\
            card_fig2('Total_Sales'),card_fig2('Total_Orders'),card_fig2('Total_quantity'),card_fig2('Total_Transaction'),card_fig2('Total_Customer'),card_fig2('Total_Products') , line

#TAP-TWO###############################################################################################################################
@app.callback(
          Output('Bar_Chart','figure'),
          Input('switch','checked'),
          Input('Dropdown','value'),
          Input('segmented','value')
)
def bar_chart(on_off ,selected_column ,segment_matrice):
    if not selected_column:
         raise PreventUpdate
    global groupby_columns
    def groupby_columns(groupby_column):
            bar_data = df.groupby(groupby_column).agg(
                                                    Total_Sales = ('Sales','sum'),
                                                    Total_freight = ('freight','sum'),
                                                    Total_Orders = ('orderID','nunique'),
                                                    Total_Transaction = ('orderID','count'),
                                                    Total_Products = ('productID','nunique'),
                                                    Total_Customers = ('customerID','nunique'))\
                                                    .assign(
                                                            Pct_Total_Sales = lambda x : (x['Total_Sales'] / x['Total_Sales'].sum()) * 100 ,
                                                            Pct_Total_freight = lambda x : (x['Total_freight'] / x['Total_freight'].sum()) * 100 ,
                                                            Pct_Total_Orders = lambda x : (x['Total_Orders'] / x['Total_Orders'].sum()) * 100 ,
                                                            Pct_Total_Transaction = lambda x : (x['Total_Transaction'] / x['Total_Transaction'].sum()) * 100 ,
                                                            Pct_Total_Products = lambda x : (x['Total_Products'] / x['Total_Products'].sum()) * 100 ,
                                                            Pct_Total_Customers = lambda x : (x['Total_Customers'] / x['Total_Customers'].sum()) * 100)\
                                                    .reset_index()\
                                                    .sort_values(by = [segment_matrice] , ascending = on_off).head(10)\
                                                    .sort_values(by = [segment_matrice],ascending=False)\
                                                    .assign(
                                                            Pct_Cumulative_Sales = lambda x : x['Pct_Total_Sales'].rolling(window = len(x),min_periods=1).sum(),
                                                            Pct_Cumulative_freight = lambda x : x['Pct_Total_freight'].rolling(window = len(x),min_periods=1).sum(),
                                                            Pct_Cumulative_Orders = lambda x : x['Pct_Total_Orders'].rolling(window = len(x),min_periods=1).sum(),
                                                            Pct_Cumulative_Transaction = lambda x : x['Pct_Total_Transaction'].rolling(window = len(x),min_periods=1).sum(),
                                                            Pct_Cumulative_Products = lambda x : x['Pct_Total_Products'].rolling(window = len(x),min_periods=1).sum(),
                                                            Pct_Cumulative_Customers = lambda x : x['Pct_Total_Customers'].rolling(window = len(x),min_periods=1).sum(),
                                                            #   percentile_unitPrice_orders = lambda x : pd.qcut(x['Sales'] , 10 , [f'<{i}%' for i in np.linspace(0.1 , 1 ,10).round(2)])
                                                            )
                                                            

            return bar_data
    bardata = groupby_columns(selected_column)
    pct_matrics = [col for col in bardata.columns[~bardata.columns.isin([segment_matrice])] if segment_matrice.split('_')[1] in col][0]
    pct_matrics_cumulative = [col for col in bardata.columns[~bardata.columns.isin([segment_matrice])] if segment_matrice.split('_')[1] in col][1] 

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_bar(y=bardata[segment_matrice],
                x=bardata[selected_column],
                text=bardata[segment_matrice],
                hovertemplate="<b>%{x:.s}</b> "+" <b>$%{y:,.2f}</b>",
                name='',
                )\
                .update_traces(texttemplate="%{text:.2s}" , textposition='outside',marker_color = px.colors.qualitative.Light24,)
    fig.add_scatter(x=bardata[selected_column] , y=bardata[pct_matrics_cumulative],name='',hoverinfo='x+y',hovertemplate ="<b>Cumulative<br>%%{y:.2f}%</b>",secondary_y=True, yaxis="y2")
    fig.add_scatter(x=bardata[selected_column] , y=bardata[pct_matrics],hoverinfo='x+y',name='',hovertemplate ="<b>Pct Total<br>%%{y:.2f}%</b>",secondary_y=True, yaxis="y2")
    fig.update_layout(
                    height=600,
                    showlegend=False,
                    title_text=f'The {"Top" if on_off != True else "Less"} 10 {selected_column.split("_")[1] if "_" in selected_column else selected_column} have <b>{round(10/df[selected_column].nunique() * 100)}%</b> of Total Number Of Whole {df[selected_column].nunique()} {selected_column.split("_")[1] if "_" in selected_column else selected_column}<br>And have <b>${round(bardata[segment_matrice].sum()):,}</b> in {segment_matrice.split("_")[1]}, and <b>{round(bardata[pct_matrics].sum())}%</b> of {segment_matrice.replace("_"," ")}',
                    title_x=0.5,
                    title_font_family='Arile',
                    title_font_size=20,
                    plot_bgcolor='rgb(34, 34, 34)',
                    paper_bgcolor='rgb(34, 34, 34)',
                    yaxis=dict(visible=False),
                    xaxis=dict(visible=True),
                    yaxis2=dict(visible=False))


    return fig

@app.callback(
          Output('Pct_Area_Chart','figure'),
          Output('Line_Hover_Area','figure'),
          Output('heatmap','figure'),
          Input('Dropdown_X','value'),
          Input('Dropdown_Y','value'),
          Input('x_column_dropdown','value'),
          Input('segmented','value'),
          Input('Dropdown','value'),
          Input('Pct_Area_Chart','hoverData'),
          Input('chips','value')
)
def area_line(X, Y , x_column , segment_matrice ,selected_column ,hover_filter,top_less):

    if not selected_column and hover_filter :
        raise PreventUpdate
    # if not X and Y:
    #      raise PreventUpdate


    if segment_matrice == 'Total_Sales':
        area_data = df.pivot_table(index=x_column,columns=selected_column,values=['Sales'], aggfunc=['sum']).apply(lambda x : (x / sum(x))*100 ,axis=1).droplevel([0,1],axis=1).reset_index()
        heatMap_data = df.pivot_table(index=X , columns=Y , values=['Sales'], aggfunc=['sum']).droplevel([0,1],axis=1).reset_index()
        top10 = df.groupby([selected_column])[['Sales']].sum().nlargest(10,'Sales').index.tolist()
        less10 = df.groupby([selected_column])[['Sales']].sum().nsmallest(10,'Sales').index.tolist()

    elif segment_matrice ==  'Total_freight':
        area_data = df.pivot_table(index=x_column,columns=selected_column,values=['freight'], aggfunc=['sum']).apply(lambda x : (x / sum(x))*100 ,axis=1).droplevel([0,1],axis=1).reset_index()
        heatMap_data = df.pivot_table(index=X , columns=Y , values=['freight'], aggfunc=['sum']).droplevel([0,1],axis=1).reset_index()
        top10 = df.groupby([selected_column])[['freight']].sum().nlargest(10,'freight').index.tolist()
        less10 = df.groupby([selected_column])[['freight']].sum().nsmallest(10,'freight').index.tolist()

    elif  segment_matrice == 'Total_Transaction':
        area_data = df.pivot_table(index=x_column,columns=selected_column,values=['orderID'], aggfunc=['count']).apply(lambda x : (x / sum(x))*100 ,axis=1).droplevel([0,1],axis=1).reset_index() 
        heatMap_data = df.pivot_table(index=X , columns=Y , values=['orderID'], aggfunc=['count']).droplevel([0,1],axis=1).reset_index()
        top10 = df.groupby([selected_column])[['orderID']].count().nlargest(10,'orderID').index.tolist()
        less10 = df.groupby([selected_column])[['orderID']].count().nsmallest(10,'orderID').index.tolist()  

    elif  segment_matrice == 'Total_Orders':
        area_data = df.pivot_table(index=x_column,columns=selected_column,values=['orderID'], aggfunc=[lambda x : x.nunique()]).apply(lambda x : (x / sum(x))*100 ,axis=1).droplevel([0,1],axis=1).reset_index() 
        heatMap_data = df.pivot_table(index=X , columns=Y , values=['orderID'],aggfunc=[lambda x : x.nunique()]).droplevel([0,1],axis=1).reset_index()
        top10 = df.groupby([selected_column])[['orderID']].nunique().nlargest(10,'orderID').index.tolist()
        less10 = df.groupby([selected_column])[['orderID']].nunique().nsmallest(10,'orderID').index.tolist()   

    elif  segment_matrice == 'Total_Products':
        area_data = df.pivot_table(index=x_column,columns=selected_column,values=['productID'], aggfunc=[lambda x : x.nunique()]).apply(lambda x : (x / sum(x))*100 ,axis=1).droplevel([0,1],axis=1).reset_index() 
        heatMap_data = df.pivot_table(index=X , columns=Y , values=['productID'], aggfunc=[lambda x : x.nunique()]).droplevel([0,1],axis=1).reset_index()
        top10 = df.groupby([selected_column])[['productID']].nunique().nlargest(10,'productID').index.tolist()
        less10 = df.groupby([selected_column])[['productID']].nunique().nsmallest(10,'productID').index.tolist()  

    elif  segment_matrice == 'Total_Customers':
        area_data = df.pivot_table(index=x_column,columns=selected_column,values=['customerID'], aggfunc=[lambda x : x.nunique()]).apply(lambda x : (x / sum(x))*100 ,axis=1).droplevel([0,1],axis=1).reset_index() 
        heatMap_data = df.pivot_table(index=X , columns=Y , values=['customerID'], aggfunc=[lambda x : x.nunique()]).droplevel([0,1],axis=1).reset_index()
        top10 = df.groupby([selected_column])[['customerID']].nunique().nlargest(10,'customerID').index.tolist()
        less10 = df.groupby([selected_column])[['customerID']].nunique().nsmallest(10,'customerID').index.tolist()              
    else:
         pass
    
    

    if top_less  == 'Top 10':
        area_data = area_data[[x_column,*[col for col in area_data.columns if col in top10]]]


    elif top_less == 'Less 10':
        area_data = area_data[[x_column,*[col for col in area_data.columns if col in less10]]]

    else:
        area_data

   
    heatMap =px.imshow(heatMap_data.set_index(X), aspect="auto", origin='lower')\
                        .update_traces(text = heatMap_data.iloc[:,1:].values , texttemplate="%{text:,.2s}")\
                        .update_layout(coloraxis_showscale=False,height=630,title=f'<b>{segment_matrice.replace("_"," ")} By {X.replace("_"," ") if "_" in X else X} & {Y.replace("_"," ") if "_" in Y else Y}</b>',
                                       paper_bgcolor = 'rgb(34, 34, 34)',
                                       xaxis_title = '' ,yaxis_title = '',title_x=0.5,title_font_size=20 , title_font_family='Arile')

                         
    
    area = px.area(
                area_data,
                x=x_column,
                y=area_data.columns[1:],
                markers=True,
                labels={'variable':area_data.columns.names[0].replace("_"," "),'value':'Pct Total'},
                hover_data={'value':':.2f'},
                custom_data=['variable'],

                )\
                .update_traces(line = dict(width=0.5) , marker = dict(size=4))\
                .update_layout(
                    showlegend=False,
                    xaxis_type='category',
                    yaxis=dict(
                        type='linear',
                        ticksuffix='%',
                     ),
                     plot_bgcolor='rgb(34, 34, 34)',
                    paper_bgcolor='rgb(34, 34, 34)',
                    title=dict(text=f'<b>Pct Of {segment_matrice.replace("_"," ")} For Each {x_column.replace("_"," ")} By {selected_column.replace("_"," ")}</b>',font_size=20,x=0.5,font_family='Arile'))
    
    fillter = hover_filter['points'][0]['customdata'][0]

    line_hover_data = df[df[selected_column] == fillter].groupby('orderDate').agg(
                                                                            Total_Sales = ('Sales','sum'),
                                                                            Total_freight = ('freight','sum'),
                                                                            Total_Orders = ('orderID','nunique'),
                                                                            Total_Transaction = ('orderID','count'),
                                                                            Total_Products = ('productID','nunique'),
                                                                            Total_Customers = ('customerID','nunique'))\
                                                                            .reset_index()
          
         
    line_hover = px.line(
                        line_hover_data,
                        x='orderDate',
                        y= segment_matrice,
                        title=f'<b>{segment_matrice.replace("_"," ")} By {selected_column.replace("_"," ")} = {fillter}<br>',
                        line_shape='spline',
                        labels={selected_column.replace("_"," ") : ''})\
                        .update_layout(xaxis={'showgrid':False,'rangeslider_visible':True},
                                       yaxis={'showgrid':False,'tickformat':'2s','ticks':"outside",'title':''},
                                        plot_bgcolor='rgb(34, 34, 34)',
                                        paper_bgcolor='rgb(34, 34, 34)',
                                        title_x = 0.5,
                                        title_font_size=20,
                                        title_font_family='Arile',
                                        height=500)
    





    return area ,line_hover, heatMap
    
#TAP-THERE#############################################################################################################################

@app.callback(
     Output('scatter_chart','figure'),
     Input('groupby_columns','value'),
     Input('x_column','value'),
     Input('y_column','value'),
     Input('color_column','value'),
     Input('size_column','value')
)        
def scatter_plot(groupby_columns,x , y , color , size):

    scatter_data = get_data_matrices(groupby_columns)

    fig =  px.scatter(
            scatter_data,
            x=x,
            y=y,
            size=size,
            color=color,
            custom_data=[groupby_columns],
            hover_name=groupby_columns,
            labels={
                    x:x.replace('_',' '),
                    y:y.replace('_',' '),
                    size:size.replace('_',' '),
                    color:color.replace('_',' ')},
            hover_data={x:':.2s',
                        size:':.2f',
                        color:':.2f',
                        },
            title=f"<b>{x.replace('_',' ')} & {y.replace('_',' ')} By {groupby_columns.replace('_',' ') if '_' in groupby_columns else groupby_columns}<br>Color: By {color.replace('_',' ')}<br>Size: By {size.replace('_',' ')}</b>"            
            )\
            .update_traces(marker_line = dict(width=1,color='black') 
                        #    , opacity=0.5,
                        )\
            .update_layout(xaxis=dict(showgrid=False),
                           yaxis=dict(showgrid=False),
                           plot_bgcolor ='rgb(34, 34, 34)',
                           paper_bgcolor = 'rgb(34, 34, 34)',)
    return fig 


@app.callback(
    Output('rank_1','figure'),
    Output('rank_2','figure'),
    Output('rank_3','figure'),
    Output('rank_4','figure'),
    Output('rank_5','figure'),
    Output('rank_6','figure'),
    Output('rank_7','figure'),
    Output('rank_8','figure'),
    Output('rank_9','figure'),
    Output('rank_10','figure'),
    Output('rank_11','figure'),
    Output('rank_12','figure'),
    Output('rank_13','figure'),
    Output('rank_14','figure'),
    Output('rank_15','figure'),
    Output('rank_16','figure'),
    Output('rank_17','figure'),
    Output('rank_18','figure'),
    Output('rank_19','figure'),
    Output('rank_20','figure'),
    Input('scatter_chart','hoverData'),
    Input('groupby_columns','value')

)
def Ranks(hover,groupby_column):

    global ranks_data
    ranks_data = get_data_matrices(groupby_column)

    selected_value = hover['points'][0]['customdata'][0]

    def get_rank_fig(rank_by):

        df = ranks_data.sort_values(by=rank_by,ascending=False).assign(rank_vales = lambda x : range(1,len(x)+1))

        value = df.loc[df[groupby_column] == str(selected_value),'rank_vales'].min()
    
        fig = go.Figure(go.Indicator(
                        mode = "number",
                        value = value,
                        number = {'font_size':40 }
                        ))\
                    .update_layout(title = f'{rank_by.replace("_"," ")}',title_x = 0.5 ,title_font_size = 13 ,
                            font_family='Raleway',plot_bgcolor='rgb(34, 34, 34)',paper_bgcolor = 'rgb(34, 34, 34)')
        return fig
    

    return get_rank_fig("Total_Sales"),get_rank_fig("Total_Orders"),get_rank_fig("Total_quantity"),get_rank_fig("Average_Sales"),get_rank_fig("Total_Transaction"),get_rank_fig("Total_discontinued_orders"),\
           get_rank_fig("Average_Orders"),get_rank_fig("life_value_day"),get_rank_fig("life_from_last_order"),get_rank_fig("Average_Sales_Per_Order"),get_rank_fig("Average_Sales_Per_Customer"),get_rank_fig("Average_Sales_Per_Product"),\
           get_rank_fig("Average_Orders_Per_Customer"),get_rank_fig("Average_Orders_Per_Product"),get_rank_fig("Average_Sales_Per_Day"),get_rank_fig("Average_Order_Per_Day"),get_rank_fig("Average_Customer_Per_Day"),get_rank_fig("Average_Product_Per_Day"),\
           get_rank_fig("Average_Sales_Per_Month"),get_rank_fig("Average_Sales_Per_Quarter"),

#TABFOUR#####################################################################################################################################################
@app.callback(
     Output('line_scenario','figure'),
     Output('Card_Senario_1','figure'),
     Output('Card_Senario_2','figure'),
     Output('Card_Senario_3','figure'),
     Output('Card_Senario_4','figure'),
     Input('slider_scenario','value'),
     Input('Scenario_dropdown','value'),
     Input('Scenario_Multi_Products','value')
)
# Scenario_dropdown
# Scenario_Multi_Products
def scenarios(slider,date_matrice,products):
    # if not products:
    #     raise PreventUpdate
     
    senario_data = df.assign(   Actual_Sales = df.quantity * df.unitPrice_products ,
                                Scenario_Sales = df.quantity * (df.unitPrice_products * (1 + slider)),
                                Actual_Vs_Scenario = lambda x : x['Scenario_Sales'] - x['Actual_Sales'],
                                Pct_Change_Pricing_Scenario = lambda x : x['Actual_Vs_Scenario'] / x['Actual_Sales'],
                                )
    if not products:
        senario_data = senario_data\
                        .groupby(date_matrice)\
                        .agg(Total_Actual_Sales =  ('Actual_Sales','sum'),
                            Total_Scenario_Sales = ('Scenario_Sales','sum'),
                            Total_Actual_Vs_Scenario = ('Actual_Vs_Scenario','sum'),
                            Total_Pct_Change_Pricing_Scenario = ('Pct_Change_Pricing_Scenario','sum'))\
                        .reset_index()
        Average_Sales_Products =    senario_data['Total_Actual_Sales'].sum() / df.productName.nunique()
        ave_senario_sales_products = senario_data['Total_Scenario_Sales'].sum() / df.productName.nunique()
        title = f'Total Actual Average Sales For All Products : <b>{Average_Sales_Products:.2f}</b><br>Total Scenario Average Sales For All Products : <b>{ave_senario_sales_products:.2f}</b>'
    else:
        senario_data = senario_data\
                        .query('productName in @products')\
                        .groupby(date_matrice)\
                        .agg(Total_Actual_Sales =  ('Actual_Sales','sum'),
                            Total_Scenario_Sales = ('Scenario_Sales','sum'),
                            Total_Actual_Vs_Scenario = ('Actual_Vs_Scenario','sum'),
                            Total_Pct_Change_Pricing_Scenario = ('Pct_Change_Pricing_Scenario','sum'))\
                        .reset_index()
        
        Average_Sales_Products =    senario_data['Total_Actual_Sales'].sum() / df.productName.nunique()
        ave_senario_sales_products = senario_data['Total_Scenario_Sales'].sum() / df.productName.nunique()
        title = f'Total Actual Average Sales For Selected Products : <b>{Average_Sales_Products:.2f}</b><br>Total Scenario Average Sales For Selected Products : <b>{ave_senario_sales_products:.2f}</b>'
        
    fig = go.Figure() 

    fig.add_scatter(
                      x = senario_data[date_matrice],
                      y= senario_data.Total_Actual_Sales,
                      name='Total Actual Sales',
                      line=dict(shape='spline'),
                      hovertemplate='<br>%{y:.2s}'
          )
    fig.add_scatter(
                      x = senario_data[date_matrice],
                      y= senario_data.Total_Scenario_Sales,
                      name='Total Scenario Sales',
                      line=dict(shape='spline',dash='dot'),
                       hovertemplate='<br>%{y:.2s}'
          )          
                    

    fig.update_layout(
                      title = title,
                      title_font_family='Arile',
                      title_font_size=20,
                      height=600,
                      hovermode='x unified',
                      plot_bgcolor='rgb(34, 34, 34)',
                      paper_bgcolor='rgb(34, 34, 34)',
                      xaxis={'showgrid':False ,'title':date_matrice.replace('_','') if '_' in date_matrice else date_matrice, 'ticks':'outside'
                            #  ,'rangeslider_visible':True
                             },
                      yaxis={'showgrid':False ,'title':'Total Sales','tickformat':'2s'},
                      legend={'x':0.9,'y':1.2}) 
            
    card_1 =go.Figure(go.Indicator(
                    mode = "number",
                    value =senario_data['Total_Actual_Sales'].sum(),
                    number = {'valueformat': ",.2s" ,'font_size':40 }
                    ))\
                .update_layout(title = f'Total Actual Sales',title_x = 0.5  , title_font_size=20,
                        font_family='Arile',plot_bgcolor='rgb(34, 34, 34)',paper_bgcolor = 'rgb(34, 34, 34)')
    
    card_2 =go.Figure(go.Indicator(
                    mode = "number",
                    value =senario_data['Total_Actual_Vs_Scenario'].sum(),
                    number = {'font_size':40 }
                    ))\
                .update_layout(title = f'Actual Vs Scenario',title_x = 0.5  , title_font_size=20,
                        font_family='Arile',plot_bgcolor='rgb(34, 34, 34)',paper_bgcolor = 'rgb(34, 34, 34)')
    
    card_3 =go.Figure(go.Indicator(
                    mode = "number",
                    value =senario_data['Total_Scenario_Sales'].sum(),
                    number = {'valueformat': ",.2s" ,'font_size':40 }
                    ))\
                .update_layout(title = f'Total Scenario Sales',title_x = 0.5  , title_font_size=20,
                        font_family='Arile',plot_bgcolor='rgb(34, 34, 34)',paper_bgcolor = 'rgb(34, 34, 34)')
    
    card_4 =go.Figure(go.Indicator(
                    mode = "number",
                    value =senario_data['Total_Pct_Change_Pricing_Scenario'].sum(),
                    number = {'font_size':40 }
                    ))\
                .update_layout(title = f'% Change Pricing Scenario',title_x = 0.5  , title_font_size=20,
                        font_family='Arile',plot_bgcolor='rgb(34, 34, 34)',paper_bgcolor = 'rgb(34, 34, 34)')
       
    
    return fig , card_1 , card_2 , card_3 , card_4


if __name__ == '__main__':
    app.run_server(port=8888 , debug = False)



































