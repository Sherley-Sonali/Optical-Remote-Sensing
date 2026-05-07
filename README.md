# Spatial Downscaling of MODIS Evapotranspiration Using High-Resolution Satellite Features and Deep Learning (Vision Transformer)

## Step 1: ET Modeling

1. Run the two JavaScript files in the `ET_Modeling` folder using the Google Earth Engine (GEE) Code Editor:

   * `Hyderabad_ET_Feature_Extraction.js`
   * `Miryalaguda_Seasonal_ET_Feature_Extraction.js`

2. After execution, export tasks will appear in the **Tasks** tab of the GEE Code Editor.

3. Run and download the generated CSV files:

   * `Hyderabad_ET_training.csv`
   * `Miryalaguda_ET_Kharif_training.csv`
   * `Miryalaguda_ET_Rabi_training.csv`

4. These CSV files are then used as training datasets for the deep learning model.
