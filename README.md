# 🌤️ Daily Maximum Temperature Forecasting in Southern Vietnam

A comprehensive machine learning project for predicting maximum temperature in Southern Vietnam using weather data from 2015-2025.

This project is part of the ADY201m course at FPT University and is implemented in Python with a focus on a reproducible ML workflow (data → features → model → evaluation → artifacts).

## 🌟 Overview

This project implements a complete machine learning pipeline to predict maximum daily temperature (`tempmax`) in Southern Vietnam using historical weather data. The project includes data preprocessing, feature engineering, model training, hyperparameter optimization, and prediction capabilities.

### 🎯 Objectives

- Predict maximum daily temperature with high accuracy
- Compare multiple ML algorithms (Random Forest, XGBoost, LightGBM, Hist-Gradient Boosting)
- Implement robust data preprocessing and feature engineering
- Create a production-ready prediction system
- Provide comprehensive model evaluation and visualization

## ✨ Features

- **📊 Comprehensive Data Analysis**: Exploratory data analysis with 70,000+ weather records
- **🔧 Advanced Preprocessing**: Missing value handling, outlier detection, data quality assessment
- **⚡ Feature Engineering**: Temporal features, lag features, rolling statistics, seasonal patterns
- **🤖 Multiple ML Models**: Random Forest, XGBoost, Decision Tree, Gradient Boosting
- **🎛️ Hyperparameter Optimization**: Automated tuning using Optuna
- **📈 Model Evaluation**: Comprehensive metrics (MAE, RMSE, R²)
- **📊 Visualization**: Model comparison charts, performance metrics, data insights
- **🚀 Production Ready**: Modular scripts, configuration management, logging
- **📝 Documentation**: Complete documentation and usage examples

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip or uv

### Setup
```bash
# create and activate venv (optional)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install deps (using uv / or pip)
uv pip install -r requirements.txt
```

### Run the pipeline
```bash
python -m src.train  # or the main training script in src/
```

## 📊 Results

### Model Performance


### Key Insights

- **Best Model**: XGBoost with optimized hyperparameters
- **Accuracy**: 60% R² score on test data
- **Error**: Mean Absolute Error of 0.98°C
- **Features**: Temperature, humidity, cloud cover, and solar radiation are most important
- **Seasonal Patterns**: Strong seasonal effects captured in the model

### Generated Visualizations

- Model comparison charts
- Feature importance plots
- Correlation heatmaps
- Time series analysis
- Performance metrics visualization

## 🔧 Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_data_pipeline.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Models

1. Add model configuration to `config/model_config.yaml`
2. Update `ModelTrainer` class in `src/models/train_model.py`
3. Add model class to the model registry

## 📈 Performance Monitoring

The project includes comprehensive performance monitoring:

- **Memory Usage**: Track memory consumption during training
- **Execution Time**: Monitor processing time for each step
- **Model Metrics**: Detailed performance metrics for all models
- **Logging**: Comprehensive logging for debugging and monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Lê Hoài Thanh Quang** - *Initial work* - [TQuang122](https://github.com/TQuang122)

## 🙏 Acknowledgments

- FPT University for providing the course framework
- Weather data sources for the dataset
- Open source ML libraries (scikit-learn, XGBoost, pandas)
- The Python data science community

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/TQuang122/Daily_Maximum_Temperature_Forecasting_in_Southern_of_VietNam/issues) page
2. Create a new issue with detailed description
3. Contact the maintainers

---
