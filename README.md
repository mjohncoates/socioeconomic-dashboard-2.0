# Socioeconomic global health inequality dashboard <br>
![DASHBOARD IMAGE](https://github.com/mjohncoates/socioeconomic-dashboard-2.0/assets/167465041/da2b1980-00ac-48c5-bbf0-405c25a198ae)
Ensure all files are within the same folder to successfully run the code & dashboard, & to run <b>"pip install -r requirements.txt"</b>
## 
1. "<b>Dashboard Jupyter notebook.ipynb</b>" is the jupyter notebook file used in this project. (Recommended to use for further exploration of the code & data, is able to run dash)
2. "<b>Dashboard_py.py</b>" is the python script used at https:/johncoates.pythonanywhere.com
3. "<b>child_mortality2.csv</b>" is the dataset that contains child mortality data.
4. "<b>childmortimage2.png</b>" is an image I created to represent the ratio of deaths between poor and rich countries. It is used within the dashboard.
5. "<b>exeterbuswhite.png</b>" is an image containing The University of Exeter Business School logo, which is used in the header of the dashboard.
6. "<b>gdp_data_csv.csv</b>" contains GDP data for countries and continents around the world overtime.
7. "<b>life-expectancy-vs-gdp-per-capita.csv</b>" contains life expectancy and gdp data for countries across the world overtime.
8. "<b>requirements.txt</b>" contains the libraries needed to run this code
## Dataset sources:<br>
<b>1.</b> life-expectancy-vs-gdp-per-capita.csv contained GDP per capita values and Life expectancy values for countries across the globe overtime. This was obtained from: <br>
https://ourworldindata.org/grapher/life-expectancy-vs-gdp-per-capita <br> <br>
<b>2.</b> child_mortality2.csv contained child mortality rate (under-5 per 1000) for countries across the world overtime. This was obtained from: <br>
https://data.worldbank.org/indicator/SP.DYN.IMRT.IN <br> <br>
<b>3.</b> gdp_csv_data.csv contains GDP values for countries and continents across the world. This was used to supplement the GDP data where it was missing. This was obtained from: <br>
https://www.imf.org/external/datamapper/NGDPDPC@WEO/OEMDC/ADVEC/WEOWORLD <br>
### Note:
1. The ocenia and asia GDP values (on the scatter plot) are misrepresented. They share GDP per capita values. The life-expectancy value is not represented for these variables.
2. If the webpage isn't displaying correctly, try zooming out on your browser.
### Future improvements:
1. More up to date data
2. Add ability to change data for scatter plot, so can look at child mortality vs gdp overtime as well.
3. More variables e.g poverty classification, access to tap water, rate of malnutrition, access to healthcare etc.
