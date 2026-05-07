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
### MODIS ET Downscaling - Random Forest & XGBoost

Spatial downscaling of MODIS MOD16A2 Evapotranspiration (500 m → 10–30 m) using ensemble machine learning models trained on Sentinel-2, Landsat 8, and SRTM features.

**Study Areas:** Hyderabad (annual) | Miryalaguda Kharif & Rabi seasons, Telangana, India



### Datasets Required

Place these CSVs in the same folder before running:

| File | Features | Samples |
|------|----------|---------|
| `Hyderabad_ET_training.csv` | NDVI, NDWI, EVI, Albedo, LST, NIR_RED_ratio, SWIR_NIR_ratio, Elevation, Slope | 12,789 |
| `Miryalaguda_ET_Kharif_training.csv` | Same + LSWI | ~9,500 |
| `Miryalaguda_ET_Rabi_training.csv` | Same + LSWI | ~8,500 |

Target variable in all three: `MODIS_ET` (mm/8-day)



### How to Run

```bash
pip install pandas numpy matplotlib scikit-learn xgboost joblib
python3 et_downscaling_RF_XGB.py
```

Outputs saved to `./outputs/` automatically.


## Results

| Dataset | Model | RMSE | R² |
|---------|-------|------|----|
| Hyderabad Annual | RF | 1.034 | 0.361 |
| Hyderabad Annual | XGB | 1.027 | 0.370 |
| Miryalaguda Kharif | RF | 1.277 | 0.228 |
| Miryalaguda Kharif | XGB | 1.265 | 0.243 |
| Miryalaguda Rabi | RF | 1.558 | 0.465 |
| Miryalaguda Rabi | XGB | 1.552 | 0.470 |

XGBoost marginally outperforms Random Forest across all datasets. LST and NDWI are the dominant ET predictors in the annual and dry-season datasets; Elevation and LSWI take over during the Kharif monsoon season.
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


## CNN-Based Evapotranspiration Estimation

A deep learning project for estimating evapotranspiration (ET) using a 1D Convolutional Neural Network (CNN) implemented in PyTorch. The model predicts MODIS-derived ET values from meteorological and environmental input parameters for different agricultural seasons and regions.

---

## Overview
- Data preprocessing and cleaning
- Feature scaling
- CNN model training with early stopping
- Model evaluation using regression metrics
- Visualization of prediction performance and residual analysis

The implementation supports datasets for:

- Kharif season
- Rabi season
- Hyderabad region

---

## Features

- 1D CNN architecture using PyTorch
- Batch Normalization and Dropout regularization
- Early stopping to prevent overfitting
- Multiple evaluation metrics:
  - RMSE
  - R² Score
  - MAE
  - Bias
- Automatic plot generation
- Correlation heatmap visualization
- Modular dataset selection

---

## Model Architecture

The CNN architecture consists of:

```text
Input
  ↓
Conv1D (1 → 16, kernel=3)
  ↓
Batch Normalization
  ↓
ReLU
  ↓
Conv1D (16 → 32, kernel=2)
  ↓
Batch Normalization
  ↓
ReLU
  ↓
Flatten
  ↓
Fully Connected Layer (64 neurons)
  ↓
Dropout (0.3)
  ↓
Output Layer (1 neuron)
```

---

## Technologies Used

- Python, PyTorch, NumPy, Pandas, Matplotlib, and Scikit-learn.

---


## Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib scikit-learn
```

---

## Dataset Configuration

Inside the script, select the dataset:

```python
dataset_name = "rabi"
```

Available options:

```python
"kharif"
"rabi"
"hyderabad"
```

---

## Data Preprocessing

The preprocessing pipeline includes:

- Removal of unnecessary columns
- Train-validation-test split:
  - 70% Training
  - 15% Validation
  - 15% Testing
- Feature normalization using `StandardScaler`
- Reshaping input for Conv1D compatibility

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | MSE Loss |
| Epochs | 200 |
| Batch Size | 128 |
| Early Stopping Patience | 15 |

---

## Evaluation Metrics

### Root Mean Squared Error (RMSE)


---

### Mean Absolute Error (MAE)

---

### Coefficient of Determination (R²)

---

### Bias


---

## Generated Plots

The script automatically generates and saves:

1. Training Loss Curve
2. Actual vs Predicted Scatter Plot
3. Residual Plot
4. Error Distribution Histogram
5. Correlation Heatmap
6. Prediction vs Sample Index Plot

Plots are saved under:

```text
plots/<dataset_name>/
```

# Key Improvements Included

Compared to a basic CNN implementation, this model includes batch normalization, improved dropout regularization, proper feature scaling without data leakage, additional evaluation metrics, enhanced plot labeling with units, correlation analysis, and an early stopping mechanism.

---
