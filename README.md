<h1 align="center">DAILY MAXIMUM TEMPERATURE FORECASTING IN SOUTHERN VIETNAM</h1>

<p align="center"><i>Predicting Tomorrow’s Heat, Today’s Innovation</i></p>

<p align="center">
  <!-- last commit -->
  <img src="https://img.shields.io/github/last-commit/TQuang122/Daily_Maximum_Temperature_Forecasting_in_Southern_of_VietNam?style=for-the-badge" />
  <!-- giả lập tỷ lệ notebook -->
  <img src="https://img.shields.io/badge/jupyter%20notebook-99.8%25-blue?style=for-the-badge" />
  <!-- số ngôn ngữ -->
  <img src="https://img.shields.io/github/languages/count/TQuang122/Daily_Maximum_Temperature_Forecasting_in_Southern_of_VietNam?style=for-the-badge" />
</p>

<p align="center"><i>Built with the tools and technologies:</i></p>

<p align="center">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/matplotlib-11557C?style=for-the-badge" />
  <img src="https://img.shields.io/badge/seaborn-4C72B0?style=for-the-badge" />
  <img src="https://img.shields.io/badge/uv-9900FF?style=for-the-badge" />
</p>

## Overview

Daily_Maximum_Temperature_Forecasting_in_Southern_of_VietNam is a comprehensive machine learning pipeline tailored for accurate weather prediction in Southern Vietnam. It streamlines data preprocessing, feature engineering, model training, and evaluation within a reproducible framework, enabling developers and students to build robust forecasting systems.

**Why Forecast Maximum Temperature in Southern Vietnam?**

This project aims to deliver precise temperature forecasts by leveraging advanced ML techniques and good MLOps practices. The core features include:

- 🧪 **Model Comparison & Tuning**: Supports multiple algorithms like XGBoost, LightGBM, RandomForest, and Gradient Boosting, with hyperparameter optimization for optimal performance.
- ⚙️ **Reproducible Pipelines**: Automates data processing, feature engineering, and evaluation, ensuring consistency across experiments.
- 📊 **Insightful Visualizations**: Provides utilities for performance analysis, feature importance, and model diagnostics.
- 🚀 **Deployment-Ready Artifacts**: Includes pre-trained models and metadata for seamless integration into production or demo environments.
- ⏱️ **Time Series Validation**: Implements date-aware cross-validation tailored for temporal data, enhancing model reliability.
- 🛠️ **Monitoring & Evaluation**: Offers tools for performance tracking and resource profiling to optimize workflows.

![overview_pipeline](./figures/overview_pipeline.png)

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



## 📊 Results

### Model Performance

![output](./figures/output.png)

### Key Insights

- **Best Model**: Random Forest with optimized hyperparameters
- **Accuracy**: 60% R² score on test data
- **Error**: Mean Absolute Error of 0.98°C
- **Features**: Temperature, humidity, cloud cover, and solar radiation are most important
- **Seasonal Patterns**: Strong seasonal effects captured in the model


## 📈 Performance Monitoring

The project includes comprehensive performance monitoring:

- **Memory Usage**: Track memory consumption during training
- **Execution Time**: Monitor processing time for each step
- **Model Metrics**: Detailed performance metrics for all models
- **Logging**: Comprehensive logging for debugging and monitoring

![execution_time](./figures/execution_time.png)
![memory](./figures/memory.png)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Lê Hoài Thanh Quang SE190062** - *Initial work* - [TQuang122](https://github.com/TQuang122)
- **Thái Việt Nam SE192065**
- **Nguyễn Tài Phúc SE191139**
- **Vũ Thanh Hoà SE190222**
## 🙏 Acknowledgments

- FPT University for providing the course framework
- Visual Crossing Weather data sources for the dataset
- Open source ML libraries (scikit-learn, XGBoost, pandas)
- The Python data science community

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/TQuang122/Daily_Maximum_Temperature_Forecasting_in_Southern_of_VietNam/issues) page
2. Create a new issue with detailed description
3. Contact the maintainers

---
