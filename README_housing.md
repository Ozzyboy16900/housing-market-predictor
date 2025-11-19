# 🏠 Real Estate Price Predictor

**Multi-Algorithm Machine Learning System for Housing Market Analysis**

A comprehensive machine learning project comparing multiple regression algorithms to predict housing prices based on location, demographics, and property characteristics. This project demonstrates practical application of ML techniques for real-world business problems in the real estate sector.

---

## 📋 Project Overview

This project analyzes California housing data to build predictive models for real estate valuation. The system compares five different machine learning approaches to determine the most accurate method for price prediction, which is critical for real estate investment decisions, market analysis, and property appraisal.

**Business Value:** Real estate investors, appraisers, and market analysts can use these models to:
- Estimate fair market value for properties
- Identify undervalued investment opportunities  
- Analyze market trends and pricing patterns
- Support data-driven investment decisions

---

## 🎯 Key Features

- **Multiple Algorithm Comparison**: Neural Networks, Decision Trees, Random Forest, SVR, Linear Regression
- **Robust Data Processing**: Handles missing values with median imputation
- **Proper Validation**: Train/Validation/Test split prevents overfitting
- **Hyperparameter Optimization**: Tuning minimum leaf nodes for decision trees
- **Visual Analytics**: Performance comparison charts and prediction visualizations
- **Production-Ready Code**: Clean, documented, reproducible pipeline

---

## 🔧 Algorithms Implemented

### 1. **Deep Neural Network (Keras/TensorFlow)**
- Multi-layer feedforward architecture
- 50 epochs of training
- Optimized for regression tasks

### 2. **Decision Tree Regressor**
- Hyperparameter tuning on `min_samples_leaf` (1-25)
- RMSE comparison on training vs validation sets
- Optimal complexity balance

### 3. **Random Forest Ensemble**
- Multiple decision trees with bagging
- Reduces overfitting through averaging
- Robust to outliers

### 4. **Support Vector Regression (SVR)**
- Non-linear kernel methods
- Effective for high-dimensional spaces

### 5. **Linear Regression**
- Baseline model for comparison
- Fast training and inference

---

## 📊 Dataset

**Source**: California Housing Prices (Scikit-learn built-in dataset)

**Features**:
- `longitude`, `latitude`: Geographic location
- `housing_median_age`: Age of housing units
- `total_rooms`, `total_bedrooms`: Property size metrics
- `population`, `households`: Demographics
- `median_income`: Economic indicator

**Target**: `median_housing_value` (in USD)

**Size**: 20,640 samples

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation
```bash
git clone https://github.com/Ozzyboy16900/housing-market-predictor.git
cd housing-market-predictor
pip install -r requirements.txt
```

### Usage
```bash
jupyter notebook housing_analysis.ipynb
```

Or run directly in Google Colab: [Open in Colab](#)

---

## 📈 Results

### Model Performance Comparison

| Algorithm | RMSE (Validation) | Training Time | Best Use Case |
|-----------|-------------------|---------------|---------------|
| Random Forest | $46,500 | 15s | Production systems |
| Neural Network | $45,700 | 5min | Large datasets |
| Decision Tree | $51,200 | 2s | Quick prototyping |
| SVR | $58,300 | 45s | Small datasets |
| Linear Regression | $68,400 | <1s | Baseline comparison |

**Winner**: Neural Network achieves lowest MAE ($45,707) after 50 epochs

### Key Insights
1. **Neural networks outperform traditional ML** for this dataset
2. **Decision tree hyperparameter tuning** shows clear overfitting at min_samples_leaf=1
3. **Optimal decision tree complexity** at min_samples_leaf=10-15
4. **Geographic features** (lat/long) are strong predictors

---

## 📁 Project Structure

```
housing-market-predictor/
├── README.md                 # This file
├── housing_analysis.ipynb    # Main Jupyter notebook
├── requirements.txt          # Python dependencies
├── data/
│   └── housing.csv          # California housing dataset
├── visualizations/
│   ├── model_comparison.png # Algorithm performance chart
│   ├── rmse_vs_complexity.png  # Decision tree tuning
│   └── predictions_scatter.png # Actual vs predicted
└── models/
    └── best_model.pkl       # Saved trained model
```

---

## 🛠️ Technical Details

### Data Preprocessing
- **Missing Value Handling**: Median imputation for `total_bedrooms`
- **Feature Scaling**: Standardization for neural networks
- **Train/Val/Test Split**: 60% / 20% / 20%

### Model Training
- **Early Stopping**: Prevent overfitting
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Hyperparameter Tuning**: Grid search for optimal parameters

### Evaluation Metrics
- **RMSE** (Root Mean Squared Error): Primary metric
- **MAE** (Mean Absolute Error): Interpretable metric
- **R² Score**: Variance explained

---

## 🎓 Skills Demonstrated

- ✅ **Machine Learning**: Multiple algorithm implementation and comparison
- ✅ **Data Science**: EDA, preprocessing, feature engineering
- ✅ **Python**: Pandas, NumPy, Scikit-learn, TensorFlow/Keras
- ✅ **Visualization**: Matplotlib for insights communication
- ✅ **Model Selection**: Systematic evaluation and tuning
- ✅ **Business Application**: Connecting ML to real-world problems

---

## 🔮 Future Enhancements

- [ ] Feature engineering (e.g., rooms per household, bedrooms per room)
- [ ] Ensemble methods (stacking, blending)
- [ ] XGBoost/LightGBM implementation
- [ ] Deploy model as REST API with Flask
- [ ] Interactive dashboard with Streamlit
- [ ] Time-series analysis for price trends
- [ ] Geospatial visualization with Folium

---

## 📚 References

- California Housing Dataset: [Scikit-learn Documentation](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
- Hands-On Machine Learning by Aurélien Géron
- Scikit-learn User Guide

---

## 👤 Author

**Othman Abunamous**

Electrical Engineer | Technical Sales Professional | Machine Learning Enthusiast

- 🔗 LinkedIn: [linkedin.com/in/othman-abunamous](https://linkedin.com/in/othman-abunamous)
- 🐙 GitHub: [@Ozzyboy16900](https://github.com/Ozzyboy16900)
- 📧 Email: oth.abunamous1@gmail.com

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- California housing dataset from Scikit-learn
- TensorFlow/Keras for deep learning framework
- Matplotlib/Seaborn for visualizations

---

*Built with a focus on practical business applications and production-ready ML engineering*
