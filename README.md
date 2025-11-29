<h1 align="center">DeepThermo: Time-Series Temperature Forecasting for Southern Vietnam (2015–2025)</h1>

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

## 📘 Overview

**DeepThermo** is an end‑to‑end machine learning system designed to forecast the **next-day maximum temperature** for 18 provinces across Southern Vietnam using 10 years of historical weather data (2015–2025).  
It unifies data preprocessing, feature engineering, model training, hyperparameter optimization, and evaluation into a clean, reproducible workflow suitable for both research and production.

Southern Vietnam experiences highly dynamic tropical weather, making accurate temperature forecasting essential for agriculture, energy planning, public services, and daily life.  
This project addresses that need by leveraging modern ML techniques and robust time‑series validation.

**What makes DeepThermo stand out?**

- 🧪 **Model Benchmarking & Optimization** — Evaluate and tune multiple ensemble models (XGBoost, LightGBM, Random Forest, Gradient Boosting, AdaBoost) using date‑aware cross‑validation and Optuna.
- ⚙️ **Reproducible Pipelines** — Automated workflows for preprocessing, feature engineering, model comparison, and result tracking ensure consistency across experiments.
- 📊 **Insightful Visual Analytics** — Built‑in utilities generate plots for feature importance, model diagnostics, performance curves, and error distributions.
- 🚀 **Deployment‑Ready Artifacts** — Exported models, schemas, and metadata make integration into production or FastAPI services seamless.
- ⏱️ **Time‑Series Reliability** — Implements rolling‑origin validation tailored specifically for temporal data to avoid leakage and ensure robust generalization.
- 🛠️ **Performance Monitoring** — Tracks training time, memory usage, and resource profiles throughout the pipeline.

![overview_pipeline](./figures/overview_pipeline.png)

The goal is to provide a powerful, transparent, and extensible framework for temperature forecasting — enabling developers, students, and researchers to build accurate and interpretable climate‑related ML systems.


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


### 🔍 Extended Features

- **🧩 Time‑Series Aware Splitting**  
  Implements Rolling‑Origin Cross‑Validation (ROCV) with gap, horizon, and minimum training window to prevent data leakage and improve generalization.

- **📦 Dataset Versioning & Artifacts**  
  Automatically exports:
  - Train/Val/Test splits  
  - Feature schemas  
  - Model metadata  
  - Optuna study database  
  - Final trained models  

- **🧠 Interpretability with SHAP**  
  Provides:
  - Global importance plots  
  - SHAP beeswarm  
  - Dependence plots  
  - Local explanations (force/waterfall)  

- **🛠 Modular Codebase**  
  Clearly separated modules:
  - `preprocessing/`
  - `feature_engineering/`
  - `models/`
  - `evaluation/`
  - `utils/` (rolling CV, scoring, visualization)

- **🧵 Reproducibility**  
  Full seed control for NumPy, Python, Optuna, and model libraries.

- **📡 Ready for Deployment**  
  Supports rapid integration with:
  - FastAPI endpoints  
  - Batch inference pipelines  
  - Scheduled retraining workflows  

- **🧭 Diagnostic Tools**  
  Includes:
  - Residual analysis  
  - Error distribution  
  - Province‑level performance breakdown  
  - Seasonal evaluation  
  - Prediction vs. actual overlays  

- **📈 Experiment Tracking (Optional)**  
  Native integration with Weights & Biases for:
  - Metric logging  
  - Hyperparameter sweeps  
  - Resource monitoring  
  - Artifact storage



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
