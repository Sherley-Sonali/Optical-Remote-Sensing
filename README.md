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

## Vision Transformer (ViT) Model

1. Upload the three CSV files generated in Step 1 to the Kaggle dataset:
   - `Hyderabad_ET_training.csv`
   - `Miryalaguda_ET_Kharif_training.csv`
   - `Miryalaguda_ET_Rabi_training.csv`

2. Open the notebook `ViT.ipynb` in Kaggle and update the file paths in the Configuration cell to point to your uploaded dataset.

3. Enable GPU acceleration in Kaggle: **Settings → Accelerator → GPU T4 x2**.

4. Run all cells. The notebook trains two model variants:
   - **SeasonalViT** - trained on combined Miryalaguda Kharif + Rabi data (19,472 samples) with a learnable seasonal embedding that distinguishes monsoon and dry-season ET regimes.
   - **PlainViT** - trained on Hyderabad annual data (12,789 samples) without seasonal conditioning.

### Model Performance (Test Set)

| Model | Dataset | R² | RMSE (mm/8-day) | MAE (mm/8-day) |
|---|---|---|---|---|
| SeasonalViT | Miryalaguda (Overall) | 0.8981 | 1.4833 | 1.1038 |
| SeasonalViT | Kharif (subset) | 0.2471 | 1.2913 | 0.9814 |
| SeasonalViT | Rabi (subset) | 0.4793 | 1.6531 | 1.2262 |
| PlainViT | Hyderabad | 0.3306 | 1.0393 | 0.7598 |
