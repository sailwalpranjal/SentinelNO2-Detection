# SentinelNO2-Detection

![NO₂ Detection](https://img.shields.io/badge/Sentinel-5P-blue) ![NO₂ Analysis](https://img.shields.io/badge/NO₂-Detection-brightgreen)

A Python-based project to detect and analyze atmospheric NO₂ levels using Sentinel-5P data. SentinelNO2-Detection provides efficient data processing pipelines and visualization tools for monitoring NO₂ trends, with an emphasis on studying air quality variations.

---

## 📑 Project Overview

SentinelNO2-Detection streamlines the process of accessing, transforming, and analyzing NO₂ satellite data from Sentinel-5P, enabling researchers and policymakers to gain insights into environmental conditions.

### Key Components
- **Data Processing Pipelines**: Efficiently prepares, cleans, and transforms NO₂ data for analysis.
- **Visualization Tools**: Provides functions to create visual representations and trend analyses of NO₂ data.
- **COVID-19 Analysis**: Dedicated processing for analyzing NO₂ levels.

---

## 📂 Repository Structure

```plaintext
SentinelNO2-Detection/
│
├── NB_1_Sources.ipynb            # Notebook for sourcing Sentinel-5P NO₂ data
├── NB_2_Sentinal_Tools.ipynb
├── NB_2_Sentinal_Tools.py        # Core tools for data processing
├── download_gadm.sh
└── NB_3_Processing_Pipelines_1.ipynb # End-to-end data processing and visualization
```

### Key Files

- **NB_1_Sources.ipynb**: Guide for acquiring and setting up raw NO₂ data.
- **NB_2_Sentinal_Tools.py**: Contains reusable functions for data parsing, cleaning, and transformations.
- **NB_3_Processing_Pipelines_1.ipynb**: A complete pipeline to run data preprocessing and visualization.

---

## 🔧 Requirements

Ensure you have the necessary dependencies installed.

```bash
pip install netCDF4 xmltodict geopandas
```
---

## 🚀 Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sailwalpranjal/SentinelNO2-Detection.git
   cd SentinelNO2-Detection
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt  #will be available later
   ```

3. **Setup Data**:
   Use `NB_1_Sources.ipynb` to download and organize Sentinel-5P NO₂ data.

---

## 🛠️ Usage

### 1. Data Sourcing (`NB_1_Sources.ipynb`)

Download and structure raw Sentinel-5P data for analysis. Follow the instructions in the notebook to obtain NO₂ data files.

### 2. Data Processing (`NB_2_Sentinal_Tools.py`)

Import and use the `NB_2_Sentinal_Tools.py` functions to parse and clean the data.

### 3. Pipeline Execution (`NB_3_Processing_Pipelines_1.ipynb`)

Run this notebook for end-to-end data processing. It takes the cleaned data, performs further transformations, and generates visual insights into NO₂ trends.

---

## 📊 Example Visualization

Use the following code snippets within `NB_3_Processing_Pipelines_1.ipynb` to generate visualizations of NO₂ levels:

```python
import matplotlib.pyplot as plt
import geopandas as gpd

# Example: Plotting NO₂ concentration levels on a map
def plot_no2_map(data):
    plt.figure(figsize=(10, 6))
    data.plot(column='NO2_level', cmap='viridis', legend=True)
    plt.title("NO₂ Concentration Levels")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

# Call the function with cleaned data
plot_no2_map(cleaned_data)
```

---

## 👥 Contributions

Contributions are welcome! To get involved:
- Fork the project.
- Create a new branch for each feature or bug fix.
- Open a pull request for review.

---

## 📜 License

This project is licensed under the MIT License.


