// STUDY AREA : MIRYALAGUDA (RICE FIELDS)

var miryalaguda = ee.Geometry.Rectangle([79.4, 16.7, 79.9, 17.1]);

Map.centerObject(miryalaguda, 10);
Map.addLayer(miryalaguda, {color: 'red'}, 'Miryalaguda');

// SEASON DEFINITIONS

var seasons = {
  kharif: {start: '2023-06-01', end: '2023-11-30', label: 'Kharif'},
  rabi:   {start: '2023-12-01', end: '2024-03-31', label: 'Rabi'}
};

// SENTINEL-2 CLOUD MASK

function maskS2(image) {
  var scl = image.select('SCL');
  var mask = scl.neq(3).and(scl.neq(7)).and(scl.neq(8))
               .and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
  return image.updateMask(mask);
}

// FIX 1: LANDSAT 8 CLOUD MASK

function maskLandsat(image) {
  var qa = image.select('QA_PIXEL');
  // Bit 3 = cloud shadow, Bit 4 = cloud
  var cloudMask = qa.bitwiseAnd(1 << 3).eq(0)
    .and(qa.bitwiseAnd(1 << 4).eq(0));
  return image.updateMask(cloudMask);
}

// FUNCTION: COMPUTE ALL FEATURES PER SEASON

function computeFeatures(startDate, endDate, label) {

  //--- MODIS ET ---
  var modisET = ee.ImageCollection('MODIS/061/MOD16A2')
    .filterDate(startDate, endDate)
    .select('ET')
    .mean()
    .clip(miryalaguda)
    .multiply(0.1)
    .rename('MODIS_ET');

  Map.addLayer(modisET,
    {min: 5, max: 30, palette: ['white', 'blue', 'green']},
    label + ' MODIS ET');

  //--- SENTINEL-2 ---
  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(miryalaguda)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(maskS2)
    .median()
    .clip(miryalaguda);

  var s2_scaled = s2.divide(10000);

  Map.addLayer(s2,
    {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000},
    label + ' Sentinel RGB');

  //--- FIX 2: LANDSAT 8 LST WITH CLOUD MASK ---
  var landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(miryalaguda)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUD_COVER', 20))   // pre-filter cloudy scenes
    .map(maskLandsat)                           // pixel-level cloud mask
    .median()
    .clip(miryalaguda);

  var lst = landsat.select('ST_B10')
    .multiply(0.00341802)
    .add(149.0)
    .subtract(273.15)
    .rename('LST');

  Map.addLayer(lst,
    {min: 20, max: 45, palette: ['blue', 'green', 'yellow', 'red']},
    label + ' LST');

  //--- NDVI ---
  var ndvi = s2_scaled.normalizedDifference(['B8', 'B4']).rename('NDVI');
  Map.addLayer(ndvi,
    {min: -1, max: 1, palette: ['blue', 'white', 'green']},
    label + ' NDVI');

  //--- NDWI ---
  var ndwi = s2_scaled.normalizedDifference(['B3', 'B8']).rename('NDWI');
  Map.addLayer(ndwi,
    {min: -1, max: 1, palette: ['brown', 'white', 'blue']},
    label + ' NDWI');

  //--- LSWI ---
  var lswi = s2_scaled.normalizedDifference(['B8', 'B11']).rename('LSWI');
  Map.addLayer(lswi,
    {min: -1, max: 1, palette: ['brown', 'white', 'cyan']},
    label + ' LSWI');

  //--- EVI ---
  var evi = s2_scaled.expression(
    '2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))', {
      'NIR':  s2_scaled.select('B8'),
      'RED':  s2_scaled.select('B4'),
      'BLUE': s2_scaled.select('B2')
    }).rename('EVI');

  Map.addLayer(evi,
    {min: -1, max: 1, palette: ['white', 'yellow', 'green']},
    label + ' EVI');

  //--- ALBEDO ---
  var albedo = s2_scaled.expression(
    '0.356*B2 + 0.130*B4 + 0.373*B8 + 0.085*B11 + 0.072*B12 - 0.0018', {
      'B2':  s2_scaled.select('B2'),
      'B4':  s2_scaled.select('B4'),
      'B8':  s2_scaled.select('B8'),
      'B11': s2_scaled.select('B11'),
      'B12': s2_scaled.select('B12')
    }).rename('Albedo');

  Map.addLayer(albedo,
    {min: 0, max: 0.25, palette: ['black', 'blue', 'cyan', 'yellow', 'white']},
    label + ' Albedo');

  //--- BAND RATIOS ---
  var nir_red_ratio = s2_scaled.select('B8')
    .divide(s2_scaled.select('B4'))
    .rename('NIR_RED_ratio');

  var swir_nir_ratio = s2_scaled.select('B11')
    .divide(s2_scaled.select('B8'))
    .rename('SWIR_NIR_ratio');

  Map.addLayer(nir_red_ratio,
    {min: 0, max: 5, palette: ['purple', 'cyan', 'green']},
    label + ' NIR_RED_ratio');

  Map.addLayer(swir_nir_ratio,
    {min: 0, max: 3, palette: ['yellow', 'orange', 'red']},
    label + ' SWIR_NIR_ratio');

  //--- DEM ---
  var dem = ee.Image('USGS/SRTMGL1_003')
    .clip(miryalaguda)
    .rename('Elevation');
  var slope = ee.Terrain.slope(dem).rename('Slope');

  //--- COMBINE ALL PREDICTORS ---
  var predictors = ee.Image.cat([
    ndvi, ndwi, lswi, evi, albedo,
    nir_red_ratio, swir_nir_ratio,
    lst, dem, slope
  ]);

  print(label + ' Predictor bands:', predictors.bandNames());

  //--- FIX 3: STRATIFIED RANDOM SAMPLE INSTEAD OF .sample() ---
  // Use randomPoints to get a fixed, controlled number of locations
  var samplePoints = ee.FeatureCollection.randomPoints({
    region: miryalaguda,
    points: 500,          // fixed sample size — no runaway element count
    seed: 42
  });

  //--- EXTRACT ET + PREDICTORS AT SAMPLE POINTS ---
  var etAtPoints = modisET.sampleRegions({
    collection: samplePoints,
    scale: 500,
    geometries: true
  });

  var trainingData = predictors.sampleRegions({
    collection: etAtPoints,
    scale: 30,
    geometries: true
  });

  print(label + ' Training dataset (500 pts):', trainingData.limit(5));

  //--- EXPORT TO DRIVE ---
  Export.table.toDrive({
    collection: trainingData,
    description: 'Miryalaguda_ET_' + label + '_training',
    fileFormat: 'CSV'
  });

  return predictors;
}

// RUN BOTH SEASONS

var kharif_features = computeFeatures(
  seasons.kharif.start, seasons.kharif.end, seasons.kharif.label
);

var rabi_features = computeFeatures(
  seasons.rabi.start, seasons.rabi.end, seasons.rabi.label
);

//--- DEM LAYERS (shown once) ---
var dem = ee.Image('USGS/SRTMGL1_003').clip(miryalaguda).rename('Elevation');
var slope = ee.Terrain.slope(dem).rename('Slope');

Map.addLayer(dem,   {min: 300, max: 600, palette: ['white', 'brown']}, 'Elevation');
Map.addLayer(slope, {min: 0,   max: 10,  palette: ['white', 'yellow', 'orange', 'red']}, 'Slope');