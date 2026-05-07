// STUDY AREA : HYDERABAD

var hyderabad = ee.Geometry.Rectangle([78.1,17.2,78.7,17.7]);

Map.centerObject(hyderabad,9);
Map.addLayer(hyderabad,{color:'red'},'Hyderabad');


// PHASE 1 : MODIS EVAPOTRANSPIRATION

var modis = ee.ImageCollection('MODIS/061/MOD16A2')
.filterDate('2023-01-01','2023-12-31')
.select('ET')
.mean()
.clip(hyderabad);

var modisET = modis.multiply(0.1).rename('MODIS_ET');

Map.addLayer(modisET,
{min:5,max:30,palette:['white','blue','green']},
'MODIS ET');


// SENTINEL-2 CLOUD MASK

function maskS2(image){

var scl = image.select('SCL');

var mask = scl.neq(3)
.and(scl.neq(7))
.and(scl.neq(8))
.and(scl.neq(9))
.and(scl.neq(10))
.and(scl.neq(11));

return image.updateMask(mask);

}


// SENTINEL-2 DATA

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
.filterBounds(hyderabad)
.filterDate('2023-01-01','2023-12-31')
.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))
.map(maskS2)
.median()
.clip(hyderabad);

// Convert reflectance scale (0-10000 → 0-1)

var s2_scaled = s2.divide(10000);

Map.addLayer(s2,
{bands:['B4','B3','B2'],min:0,max:3000},
'Sentinel RGB');


// LANDSAT 8 LST

var landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
.filterBounds(hyderabad)
.filterDate('2023-01-01','2023-12-31')
.median()
.clip(hyderabad);

var lst = landsat.select('ST_B10')
.multiply(0.00341802)
.add(149.0)
.subtract(273.15)
.rename('LST');

Map.addLayer(lst,
{min:20,max:45,palette:['blue','green','yellow','red']},
'LST');


// PHASE 2 : FEATURE ENGINEERING

// NDVI

var ndvi = s2_scaled.normalizedDifference(['B8','B4']).rename('NDVI');

Map.addLayer(ndvi,
{min:-1,max:1,palette:['blue','white','green']},
'NDVI');


// NDWI

var ndwi = s2_scaled.normalizedDifference(['B3','B8']).rename('NDWI');

Map.addLayer(ndwi,
{min:-1,max:1,palette:['brown','white','blue']},
'NDWI');


// EVI

var evi = s2_scaled.expression(
'2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))',{
'NIR':s2_scaled.select('B8'),
'RED':s2_scaled.select('B4'),
'BLUE':s2_scaled.select('B2')
}).rename('EVI');

Map.addLayer(evi,
{min:-1,max:1,palette:['white','yellow','green']},
'EVI');


// ALBEDO (correct reflectance inputs)

var albedo = s2_scaled.expression(
'0.356*B2 + 0.130*B4 + 0.373*B8 + 0.085*B11 + 0.072*B12 - 0.0018',{
'B2':s2_scaled.select('B2'),
'B4':s2_scaled.select('B4'),
'B8':s2_scaled.select('B8'),
'B11':s2_scaled.select('B11'),
'B12':s2_scaled.select('B12')
}).rename('Albedo');

Map.addLayer(albedo,
{min:0,max:0.25,palette:['black','blue','cyan','yellow','white']},
'Albedo');


// BAND RATIOS

var nir_red_ratio = s2_scaled.select('B8')
.divide(s2_scaled.select('B4'))
.rename('NIR_RED_ratio');

Map.addLayer(nir_red_ratio,
{min:0,max:5,palette:['purple','cyan','green']},
'NIR_RED_ratio');


var swir_nir_ratio = s2_scaled.select('B11')
.divide(s2_scaled.select('B8'))
.rename('SWIR_NIR_ratio');

Map.addLayer(swir_nir_ratio,
{min:0,max:3,palette:['yellow','orange','red']},
'SWIR_NIR_ratio');


// DEM FEATURES

var dem = ee.Image('USGS/SRTMGL1_003')
.clip(hyderabad)
.rename('Elevation');

var slope = ee.Terrain.slope(dem)
.rename('Slope');

Map.addLayer(dem,
{min:400,max:700,palette:['white','brown']},
'Elevation');

Map.addLayer(slope,
{min:0,max:10,palette:['white','yellow','orange','red']},
'Slope');


// COMBINE ALL FEATURES

var predictors = ee.Image.cat([
ndvi,
ndwi,
evi,
albedo,
nir_red_ratio,
swir_nir_ratio,
lst,
dem,
slope
]);

print('Predictor bands:', predictors.bandNames());


// SAMPLE MODIS PIXELS

var modisSamples = modisET.sample({
region: hyderabad,
scale: 500,
geometries: true
});

print('MODIS sample points', modisSamples);


// EXTRACT PREDICTORS AT SAME LOCATIONS

var trainingData = predictors.sampleRegions({
collection: modisSamples,
scale: 30,
geometries: true
});

print('Training dataset', trainingData);


// EXPORT DATASET

Export.table.toDrive({
collection: trainingData,
description:'Hyderabad_ET_training',
fileFormat:'CSV'
});