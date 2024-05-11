import pandas as pd

import plotly.graph_objs as go
from plotly import tools
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px

import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash import html, dcc, Dash, Input, Output
from dash import dcc
import plotly.express as px
import dash_bootstrap_components as dbc
import math

initial_df = pd.read_csv("child_mortality2.csv")

melted_df = pd.melt(initial_df, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
                    var_name='Year', value_name='Value')

# Pivot the dataframe to restructure it
restructured_df = melted_df.pivot_table(index=['Country Name', 'Country Code', 'Year'],
                                        columns='Indicator Name', values='Value').reset_index()

# Rename the columns as needed
restructured_df.columns.name = None  # Remove the name of the columns index
restructured_df = restructured_df.rename(columns={'Country Name': 'Entity', 'Country Code': 'Code'})

#year as integer
restructured_df['Year'] = restructured_df['Year'].astype(int)


df = pd.read_csv("life-expectancy-vs-gdp-per-capita.csv")

merged_df = pd.merge(df, restructured_df[['Entity', 'Year', 'Mortality rate, under-5 (per 1,000 live births)']], 
                     on=['Entity', 'Year'], how='left')

merged_df = merged_df.drop(0).reset_index(drop=True)

dropdown_options = [{'label': col, 'value': col} for col in df.columns]


def update_map(selected_column):
    minimum = df[selected_column].min()
    maximum = df[selected_column].max()

    fig_map = px.choropleth(df,
                            locations="Country",
                            locationmode='country names',
                            color=selected_column,
                            range_color=(minimum, maximum),
                            hover_name="Country",
                            hover_data={"Life expectancy": True, "GDP per capita": True,
                                        "Mortality rate, under-5 (per 1,000 live births)": True},
                            color_continuous_scale='Edge_r',
                            labels={'number': selected_column},
                            animation_frame='Year')

    # Update layout of the map
    fig_map.update_layout(
        title_text='Global Life Expectancy from 1950 - 2021',
        title_font=dict(size=24, color="black"),
        geo=dict(showframe=True, showcoastlines=True),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=2, r=10, t=37, b=10),  # Reduce margins
        coloraxis_colorbar=dict(len=0.75, x=0.5, y=-0.2, orientation='h'),  # Adjust position and length of colorbar
        coloraxis_colorbar_ticks="outside",  # Place colorbar ticks outside the colorbar
        coloraxis_colorbar_tickfont=dict(size=12),  # Adjust colorbar tick font size
        legend=dict(font=dict(size=10)),
        height=600, width=1000,  # Adjust plot height
        sliders=dict(
            x=0.1,  # Adjust x position of the slider 
            y=0.5,  # Adjust y position of the slider 
        )
    )

    return fig_map

# ## Figure 2 - Life expectancy vs GDP

df1 = merged_df.copy()

df1 = merged_df.copy()
# List of specified regions
specified_regions = ['Africa', 'Northen America', 'Europe', 'South America', 'Asia', 'Oceania']
# Filter rows where 'Entity' column matches specified regions and set 'Code' column to "CON"

df1.loc[df1['Entity'].isin(specified_regions), 'Code'] = "CON"
#Remove rows with empty values in column 'Code'
df1 = df1.dropna(subset=['Code'])
df1 = df1[(df1['Year'] >= 1980) & (df1['Year'] <= 2018)]
df1.rename(columns={'Entity': 'Country'}, inplace=True)
df1.rename(columns={'Period life expectancy at birth - Sex: all - Age: 0': 'Life expectancy'}, inplace=True)

# adding gdp data for continents:

gdpdata = pd.read_csv("gdp_data_csv.csv")



# Melt the DataFrame to have the years as a separate column
melted_gdpdata = pd.melt(gdpdata, id_vars=["Country"], var_name="Year", value_name="GDP_per_capita")

# Convert the "GDP_per_capita" column to float and replace "no data" with NaN
melted_gdpdata['GDP_per_capita'] = pd.to_numeric(melted_gdpdata['GDP_per_capita'], errors='coerce')

melted_gdpdata['Year'] = melted_gdpdata['Year'].astype(int)

# Sort the dataset by the "Country" columns
melted_gdpdata = melted_gdpdata.sort_values(by=['Country', 'Year']).reset_index(drop=True)

# Filter rows where "Country" column is not null
melted_gdpdata = melted_gdpdata[melted_gdpdata['Country'].notnull()]



# Reset index
melted_gdpdata.reset_index(drop=True, inplace=True)

# Display the resulting DataFrame
melted_gdpdata

###adding oceania and asia

# Duplicate rows where Country is "Asia and Pacific"
asia_pacific_rows = melted_gdpdata[melted_gdpdata['Country'] == 'Asia and Pacific'].copy()

# Replace Country with "Oceania" in the duplicated rows
asia_pacific_rows['Country'] = 'Oceania'

# Concatenate the original dataframe with the duplicated and modified rows
melted_gdpdata = pd.concat([melted_gdpdata, asia_pacific_rows], ignore_index=True)

# Duplicate rows where Country is "Asia and Pacific"
asia_pacific_rows = melted_gdpdata[melted_gdpdata['Country'] == 'Asia and Pacific'].copy()

# Replace Country with "Oceania" in the duplicated rows
asia_pacific_rows['Country'] = 'Asia'

# Concatenate the original dataframe with the duplicated and modified rows
melted_gdpdata = pd.concat([melted_gdpdata, asia_pacific_rows], ignore_index=True)

gdpcontinents = melted_gdpdata[melted_gdpdata['Country'].isin(['Africa (Region)',
                                                               'North America', 'South America',
                                                               'Asia', 'Oceania', 'Europe', 'World'])]


gdpcontinents = gdpcontinents[gdpcontinents['Year'] <= 2018]
gdpcontinents.rename(columns={'GDP_per_capita': 'GDP per capita'}, inplace=True)
gdpcontinents['Country'] = gdpcontinents['Country'].replace({'Africa (Region)': 'Africa'})


# Find unique values in the 'Country' column
unique_countries = gdpcontinents['Country'].unique()



# Merge the two DataFrames
df1 = pd.merge(df1, gdpcontinents, on=['Country', 'Year'], how='left')

# Fill missing values in 'GDP per capita' column of df1 with values from gdpcontinents
df1['GDP per capita'] = df1['GDP per capita_x'].fillna(df1['GDP per capita_y'])

# Drop the redundant columns
df1.drop(['GDP per capita_x', 'GDP per capita_y'], axis=1, inplace=True)


df1.head()

# ### Creating graph


# Create a new column indicating if each data point belongs to one of the specified regions
df1['Region'] = df1['Country'].apply(lambda x: 'Other' if x not in ['Africa', 'North America', 'Europe', 'Asia', 'Oceania', 'World'] else x)

# Create the scatter plot with different colors for the specified regions
fig_scatter = px.scatter(df1, x="GDP per capita", y="Life expectancy",
                         hover_data=["Country", "Mortality rate, under-5 (per 1,000 live births)"], hover_name="Country", animation_frame="Year",
                         trendline="ols", trendline_options=dict(log_x=True),
                         color='Region',title="GDP per capita vs. Life Expectancy Overtime",
                         color_discrete_map={'Other': 'grey'})

# Update opacity specifically for the "Other" colored dots
fig_scatter.update_traces(marker=dict(opacity=0.5), selector=dict(marker=dict(color='grey')))

# Manually set opacity for 'Other' color in color_discrete_map
fig_scatter.for_each_trace(lambda t: t.update(marker=dict(opacity=0.2, size = 4)) if t.marker.color == 'grey' else ())
# Change label in legend
fig_scatter.for_each_trace(lambda t: t.update(name='Countries') if t.marker.color == 'grey' else ())
# Manually set opacity for 'World' color in color_discrete_map
fig_scatter.for_each_trace(lambda t: t.update(marker=dict(opacity=0.8, size = 16)) if t.marker.color != 'grey' else ())

# Add title to the graph
fig_scatter.update_layout(
                            yaxis_title="Life Expectancy (years)",
                            title=dict(font_size=25),
                            xaxis_title="GDP per capita ($)",
                            legend=dict(title_font_size=20,font_size=18),
                            margin=dict(l=10, r=5, t=40, b=0),
                           yaxis=dict(title_font_size=20,tickfont_size=16),
                           xaxis=dict(title_font_size=20,tickfont_size=16),
                            xaxis_title_font=dict(size=20),
                           height=600, width=800)  # Adjust height and width of the plot



# ## Figure 3 - Bar chart showing difference in life expectancy for low income vs high income


df2 = merged_df.copy()
df2.head()


df2.rename(columns={'Entity': 'Country'}, inplace=True)
df2.rename(columns={'Period life expectancy at birth - Sex: all - Age: 0': 'Life expectancy'}, inplace=True)
years_to_keep = [2018]
df2 = df2[df2['Year'].isin(years_to_keep)]
regions_to_keep = ['Low-income countries','High-income countries']
df2 = df2[df2['Country'].isin(regions_to_keep)]
print(df2['Country'].unique())
print(df2['Year'].unique())


# Replace 'Country' with the actual column name in your DataFrame
custom_order = ['Low-income countries', 'High-income countries']

# Convert 'Country' column to categorical with custom order
df2['Country'] = pd.Categorical(df2['Country'], categories=custom_order, ordered=True)

# Sort DataFrame based on the categorical 'Country' column
df2_sorted = df2.sort_values(by='Country')

# Convert the column to string type
df2_sorted['Year'] = df2_sorted['Year'].astype(str)
# Now, df_sorted contains the DataFrame sorted according to the custom order



fig_bar_lifeexp = px.bar(df2_sorted, x="Life expectancy", y="Country", title="Life expectancy in Low-income vs. High-income countries", orientation='h')

# Update traces to hide legend
fig_bar_lifeexp.update_traces(showlegend=False)

fig_bar_lifeexp.update_layout(
    margin=dict(b=0, r=3),
                              xaxis_title_font=dict(size=18),
                             yaxis_title_font=dict(size=18),
                             xaxis=dict(tickfont=dict(size=14)),
                             yaxis=dict(tickfont=dict(size=14)),
    title_font=dict(size=20),
height = 500
)
fig_bar_lifeexp.update_xaxes(title_text="")
# Update label font size on bars
fig_bar_lifeexp.update_traces(textfont_size=24)  # Adjust size as needed
#fig_bar_lifeexp.show()


df2_sorted


fig_percent2 = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = 63,
    number ={"suffix":" Years"},
    mode = "gauge+delta+number",
    delta={'reference': 81, 'valueformat': '.0f', 'increasing.color': 'red'},
    title = {'text': "Life expectancy of low-income countries",
         'font': {'size': 25}},
    gauge = {'axis': {'range': [0, 90]},
                 'steps' : [
                 {'range': [45, 67.5], 'color': "lightgray"},
                 {'range': [67.5, 90], 'color': "gray"}],
              'threshold' : {'line': {'color': "lime", 'width': 5}, 'thickness': 0.75, 'value': 81},
             
             'bar':{'color':'red', 'thickness': 0.5}}))
fig_percent2.update_layout(
                            margin=dict(t=0, b=0),
height = 380)




fig_percent3 = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = 81,
    number ={"suffix":" Years"},
    mode = "gauge+number",
    title = {'text': "Life expectancy of high-income countries",
         'font': {'size': 25}},
    gauge = {'axis': {'range': [0, 90]},
                 'steps' : [
                 {'range': [45, 67.5], 'color': "lightgray"},
                 {'range': [67.5, 90], 'color': "gray"}],
              'threshold' : {'line': {'color': "lime", 'width': 5}, 'thickness': 0.75, 'value': 81},
             
             'bar':{'color':'lime', 'thickness': 0.5}}))
fig_percent3.update_layout(
                            margin=dict(t=0, b=0),
height=380)




# ## Figure 4 - Pictoral chart to show difference in child morality
# 


# ## Implementing child mortality statistic


# ### Find average for child mortality and word it as "x times more likely to die"


df3 = merged_df.copy()


df3.dropna(subset=['Mortality rate, under-5 (per 1,000 live births)'])


# Subset the DataFrame to include only rows where the year is 2018
df3 = df3[df3['Year'] == 2018]


df3.info()


df3_poor = df3[df3['GDP per capita'] <= 1000]
df3_poor.info()


df3_rich = df3[df3['GDP per capita'] >= 60000]
df3_rich.info()


df3_poor.head()


##average mortality for poor
average_mortality_rate_poor = df3_poor['Mortality rate, under-5 (per 1,000 live births)'].mean()
print("Average mortality rate for poor countries =", average_mortality_rate_poor)
#average mortality for rich
average_mortality_rate_rich = df3_rich['Mortality rate, under-5 (per 1,000 live births)'].mean()
print("Average mortality rate for rich countries =", average_mortality_rate_rich)


##how many times more
timesmore= average_mortality_rate_poor / average_mortality_rate_rich
print("Poor children below 5 are", timesmore, "times more likely to die than those from rich countries")


#Using Pillow to read the the image
pil_img = Image.open("childmortimage2.png")


### visualise using an image
image_path = "childmortimage2.png"
card_child_mort = dbc.Card(
    [
        dbc.CardHeader("Children are much more likely to die in low-income countries than high-income countries.",
                       style={"background-color": "Black", "color": "White", "fontSize":"30px",'border-bottom':'groove'}),
        dbc.CardBody(
            [
                html.Div(id='childmortstat', className='card-value text-center'),
                dbc.CardImg(src=pil_img, top=True)
            ]
        )
    ], style={'border':'groove'}, outline=True
)


# # Figure 5 - Different resulting life expectancy increase from same GDP increase - bar chart



# Define the coefficients and constant term
coefficient = 12.6247
constant = 21.8567


# Define the GDP per capita for which you want to predict life expectancy
gdp_per_capita_low = 2000 

# Calculate the predicted life expectancy
log_gdp_per_capita = math.log10(gdp_per_capita_low)
predicted_life_expectancy_low = coefficient * log_gdp_per_capita + constant

print("Predicted Life for 1000GDP Expectancy:", predicted_life_expectancy_low)

gdp_per_capita_lowmedium = 25000  
# Calculate the predicted life expectancy
log_gdp_per_capita = math.log10(gdp_per_capita_lowmedium)
predicted_life_expectancy_lowmedium = coefficient * log_gdp_per_capita + constant

print("Predicted Life for 10000GDP Expectancy:", predicted_life_expectancy_lowmedium)

realincrease_lowtomediumlow = predicted_life_expectancy_lowmedium - predicted_life_expectancy_low
###calculate percentage increase
lowtomedium_percentageinc = ((predicted_life_expectancy_lowmedium - predicted_life_expectancy_low) / predicted_life_expectancy_low)*100
##print results
print("Real increase of Life expectancy from increasing gdp from",gdp_per_capita_low,"to", gdp_per_capita_lowmedium,":", realincrease_lowtomediumlow)
print("Percentage increase of Life expectancy from increasing gdp from",gdp_per_capita_low,"to", gdp_per_capita_lowmedium,":", lowtomedium_percentageinc)



gdp_per_capita_medium = 48000  

# Calculate the predicted life expectancy
log_gdp_per_capita = math.log10(gdp_per_capita_medium)
predicted_life_expectancy_medium = coefficient * log_gdp_per_capita + constant

print("Predicted Life for 10000GDP Expectancy:", predicted_life_expectancy_medium)
#calculate real increase
realincrease_mediumtomediumlow = predicted_life_expectancy_medium - predicted_life_expectancy_lowmedium
###calculate percentage increase
lowmediumtomedium_percentageinc = ((predicted_life_expectancy_medium - predicted_life_expectancy_lowmedium) / predicted_life_expectancy_lowmedium)*100
##print results
print("Percentage increase of Life expectancy from increasing gdp from",gdp_per_capita_lowmedium,"to", gdp_per_capita_medium,":", realincrease_mediumtomediumlow)
print("Real increase of Life expectancy from increasing gdp from",gdp_per_capita_lowmedium,"to", gdp_per_capita_medium,":", lowmediumtomedium_percentageinc)



import pandas as pd

# Create a list of lists containing the data
data = [
    ['Income Category', 'Life expectancy increase'],  
    ['Low-income countries', 13.85],
    ['Middle-income countries', 3.58]
]

# Extract column names from the first row
columns = data[0]

# Create a DataFrame without the header
df5 = pd.DataFrame(data[1:], columns=columns)
print(df5)



import plotly.express as px

# Create the bar plot
fig_lifeexpincrease_bar = px.bar(df5, x="Income Category", y="Life expectancy increase", orientation='v',
                          title="Increase in life-expectancy when GDP per<br>capita is increased by $13000",
                          labels={"Life expectancy increase": "Estimated life expectancy increase (Years)", "Income Category": ""})



# Update title size
fig_lifeexpincrease_bar.update_layout(title_font_size=22, height=550)

# Update axis titles size and orientation
fig_lifeexpincrease_bar.update_layout(xaxis=dict(title_font_size=20, tickfont_size=20), 
                              yaxis=dict(title_font_size=20, title_standoff=20, tickangle=0, tickfont_size=20))

# Hide legend
fig_lifeexpincrease_bar.update_layout(showlegend=False)

# Update label font size on bars
fig_lifeexpincrease_bar.update_traces(textfont_size=24)  # Adjust size as needed


# ## Integrating into dashboard


import dash_bootstrap_components as dbc
# Define the dropdown options
dropdown_options = [
    {'label': 'Selected Value: Life Expectancy Overtime', 'value': 'Life expectancy'},
    {'label': 'Selected Value: Mortality Rate Under-5 Overtime', 'value': 'Mortality rate, under-5 (per 1,000 live births)'},
    {'label': 'Selected Value: GDP per Capita Overtime', 'value': 'GDP per capita'}
]



##First card
card_scatter_lifegdp = dbc.Card(
    [
        dbc.CardHeader("Life expectancy increases as GDP increases at a diminishing rate",
                       style={"background-color": "Black", "color": "White", "fontSize":"30px",'border-bottom':'groove'}),
        dbc.CardBody(
            [
                html.Div(id='global_lifeexp', className='card-value text-center'),
                dcc.Graph(figure=fig_scatter)
            ]
        )
    ], style={'border':'groove'}
)


#second card
card_gauge_3 = dbc.Card(
    [
        dbc.CardHeader('People in Low-Income Countries Die 18 Years Earlier Than those in High-Income Countries.',
                       style={"background-color": "Black", "color": "White", "fontSize": "30px",'border-bottom':'groove'}),
        dbc.CardBody(
            [
                html.Div(id='fig_gauge_3', className='card-value text-center'),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_percent3), width=6),
                    dbc.Col(dcc.Graph(figure=fig_percent2), width=6),
                ])
            ]
        )
    ], style={'border': 'groove'}, outline=True
)

##"Third" card
card_global_map = dbc.Card(
    [
        dbc.CardHeader('Socioeconomic values have increased overtime, but at an inequal rate',
                       style={"background-color": "Black", "color": "White", "fontSize":"30px",'border-bottom':'groove'}),
        dbc.CardBody(
            [
                html.Div(id='global_unemp', className='card-value text-center'),
                dbc.Row([
                    dbc.Col(html.Div("Global Socioeconomic Values Overtime", style={'font-size': '30px', 'font-weight':'bold'}), width=6),
                    dbc.Col(dcc.Dropdown(
                        id='color-dropdown',
                        options=dropdown_options,
                        clearable=False,
                        value='Life expectancy',  # Default value for the dropdown
                        style={'width': '85%', 'z-index':'1000', 'font-weight':'bold'}), width=6),
                ]),
                dcc.Graph(id='world-map')  # Use id='world-map' to reference in the callback
            ]
        )
    ], style={'border':'groove'}, outline=True
)



#"Second" card
card_figincreasebar = dbc.Card(
    [
        dbc.CardHeader("Poor countries benefit significantly more from an increase in GDP than richer countries",
                       style={"background-color": "Black", "color": "White", "fontSize":"30px",'border-bottom':'groove'}),
        dbc.CardBody(
            [
                html.Div(id='fig_lifeexpincrease_bar', className='card-value text-center'),
                dcc.Graph(figure=fig_lifeexpincrease_bar),
            ]
        )
    ], style={'border':'groove'}, outline=True
)



# ## Initalize Dash


### exeter logo
exe_logo = Image.open("exeterbuswhite.png")


# Your DataFrame 'df' should be defined here
df = merged_df.copy()
df.rename(columns={'Entity': 'Country'}, inplace=True)
df.rename(columns={'Period life expectancy at birth - Sex: all - Age: 0': 'Life expectancy'}, inplace=True)
# Remove rows with empty values in column 'Code'
df = df.dropna(subset=['Code'])
# Filter out rows where the year is below 1950
df = df[df['Year'] >= 1950]
# Drop rows where the country is 'Qatar' - qatar is an outlier and reduces visibility on graph.
df = df[df['Country'] != 'Qatar']
# Reset index after filtering
df.reset_index(drop=True, inplace=True)

# Define the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the app
app.layout = html.Div([
    html.Div([
        html.H1("Global Health Inequality Analytical Dashboard", style={'flex': '1',
                                                                               'padding': '10px 0 10px 10px',
                                                                               'text-align': 'left',
                                                                               'font-size':'64px',
                                                                               'color':'white'}),  # Set flex: 1 to the header text to push it to the center
        html.Div(html.Img(src=exe_logo, height="100px"), style={'width': '250px','background-color':'black'}),
    ], style={'display': 'flex', 'align-items': 'left', 'background-color': 'black', 'width': '100%','border-bottom':'10px solid White'}),  # Removed justify-content: center
    dbc.Row([
        dbc.Col(card_child_mort, width=6),
        #dbc.Col(card_gauge_1, width=3),
        dbc.Col(card_gauge_3, width=6)
    ]),
    dbc.Row([
        dbc.Col(card_global_map, width=5),
        dbc.Col(card_scatter_lifegdp, width=4),
        dbc.Col(card_figincreasebar, width=3)
    ])
])
df_lifeexp = df.copy()
df_gdp = df[df['Year'] <= 2018]
df_childmort = df[(df['Year'] >= 1960) & (df['Year'] <= 2018)]
# Define callback to update the map based on the dropdown selection
@app.callback(
    Output('world-map', 'figure'),
    [Input('color-dropdown', 'value')]
)

def update_map(selected_column):
    if selected_column == 'Life expectancy':
        colour='Hot'
        minimum = 40
        maximum = 90
        selected = df_lifeexp
    elif selected_column =='GDP per capita':
        colour='Blackbody'
        minimum = None
        middle = None
        maximum = None
        selected = df_gdp
    else:
        colour='Hot_r'
        minimum = 0
        middle = 50
        maximum = 100
        selected = df_childmort
    fig_map = px.choropleth(selected,
                            locations="Country",
                            locationmode='country names',
                            color=selected_column,
                            range_color=(minimum, maximum),
                            hover_name="Country",
                            hover_data={"Life expectancy": True, "GDP per capita": True,
                                        "Mortality rate, under-5 (per 1,000 live births)": True},
                            color_continuous_scale=colour,
                            labels={'number': selected_column},
                            animation_frame='Year')

    # Update layout of the map
    fig_map.update_layout(
        geo=dict(showframe=True, showcoastlines=True),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=2, r=10, t=0, b=2),  # Reduce margins
        coloraxis_colorbar=dict(len=0.75, x=0.5, y=-0.15,
                                orientation='h',outlinecolor='Black',outlinewidth=1),
        coloraxis_colorbar_ticks="outside",  # Place colorbar ticks outside the colorbar
        coloraxis_colorbar_tickfont=dict(size=12),  # Adjust colorbar tick font size
        legend=dict(
            title=dict(
                font=dict(size=22)),
        font=dict(size=22)),
    height=600, width=1000,  # Adjust plot height
        sliders=dict(
            x=0.1,  # Adjust x position of the slider (0 is left, 1 is right)
            y=0.5,  # Adjust y position of the slider (0 is bottom, 1 is top)
        )
    )
    if selected_column == 'Mortality rate, under-5 (per 1,000 live births)':
        fig_map.update_layout(
        coloraxis_colorbar=dict(
            tickmode='array',
            tickvals=[0,50,100],
            ticktext=['0','50','>100']))
    fig_map.update_layout(
    coloraxis_colorbar_tickfont=dict(size=15))

    return fig_map

