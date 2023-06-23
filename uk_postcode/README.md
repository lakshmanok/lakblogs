For more information, please see this blog post:

Working with UK geospatial data at the postcode level: How to associate UK postcodes with area census data

https://medium.com/@lakshmanok/working-with-uk-geospatial-data-at-the-postcode-level-3c9f79d866b3


There are two CSV files there:
* ukpopulation.csv.gz has the following columns:
```
postcode,latitude,longitude,area_code,area_name,all_persons,females,males
```
* ukpostcodes.csv.gz has one extra column - the polygon for each postcode in WKT format:
```
postcode,latitude,longitude,area_code,area_name,all_persons,females,males,geometry_wkt
```

In the article, I step through how I created the dataset. You can see the code in the notebook uk_postcodes.ipynb.

**NOTE:**
Use of this data and/or code is at your own risk -- this is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
