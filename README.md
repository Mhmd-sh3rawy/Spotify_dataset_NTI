# Overview

Intelligent music recommendation system that analyzes audio features to predict song popularity and provide personalized recommendations using SVR, KNN, and Deep Neural Networks.

**Key Features:** Advanced feature engineering (18+ features), mood clustering, cyclic encoding, interactive Streamlit web app.

## 📁 Project Structure

```
Spotify_dataset_NTI/
├── data/
│   ├── raw/Spotify-dataset.csv
│   └── procesed/modified-spotify-data.csv
├── deployment/
│   ├── streamlit_app.py                          # Streamlit web app
│   ├── Deployment_NoteBook.ipynb
│   ├── feature_names.pkl
│   ├── popularity_model.h5
│   ├── popularity_model.keras
│   └── scaler.pkl
├── src/
│   ├── __init__.py
│   ├── exploration.py
│   ├── plotting.py
│   └── transformation.py
├── logs/Preprocessing logs.md
├── .gitignore
├── README.md
├── 01-exploration.ipynb
├── 02-feature-analysis.ipynb
├── 03-preparing-data.ipynb
├── 04-(debolyed DNN)-Training-SVR_KNN_DNN.ipynb
├── 05-RandomForest-model.ipynb
├── 06-Training-SVR_KNN_DNN.ipynb
├── 07-Traning_Regression-clf_linear-DNN_.ipynb
├── 08.Logistic_regression_clf.ipynb
└── Spotify Presentation10.pdf
```

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/Mhmd-sh3rawy/Spotify_dataset_NTI.git
cd Spotify_dataset_NTI
pip install -r requirements.txt
```

### Dependencies
```
pandas, numpy, scikit-learn, tensorflow
matplotlib, seaborn, plotly
streamlit, jupyter, joblib
```

## 🔧 Usage

### Option 1: Streamlit Web App (Recommended)
```bash
cd deployment
streamlit run app.py
```

**Features:** Interactive UI, real-time predictions, audio feature visualization, model comparison

### Option 2: Jupyter Notebook
```bash
jupyter notebook deployment/Deployment_NoteBook.ipynb
```

### Option 3: Full Training (Google Colab Recommended)
Upload `04-(debolyed DNN)-Training-SVR_KNN_DNN.ipynb` to Google Colab for GPU-accelerated training.

## 🧠 Models & Features

### ML Algorithms
- **Support Vector Regression (SVR)** - RBF `rbf` kernels
- **K-Nearest Neighbors (KNN)** - Optimized distance metrics  
- **Deep Neural Network (DNN)** - Multi-layer with dropout & batch normalization

### Feature Engineering (18+ Features)
- **Duration Features:** Log-transformed, standardized, classified
- **Mood Clusters:** K-means on audio features
- **Interaction Features:** `happy_dance`, `acoustics_instrumental`
- **Artist Features:** Average popularity, song count
- **Cyclic Encoding:** Musical key representation
- **Audio Transformations:** Yeo-Johnson power transformation

## 📊 Results

Comprehensive model comparison with MAE, RMSE, and R² scores. DNN shows superior performance with 15-25% improvement from feature engineering.


## 🌐 Streamlit App Features

-   Manual input with sliders
-   CSV upload for batch predictions
-   Random song generator
-   Interactive visualizations
-   Real-time predictions
-   Link to share as long deployment notebook is running

## 📝 Documentation

- **Processing Logs:** `logs/Preprocessing logs.md` - Complete pipeline documentation
- **Notebooks:** Detailed markdown and code comments throughout

## 👨‍💻 Contributors

**Mohamed Sha3rawy**
- GitHub: [@Mhmd-sh3rawy](https://github.com/Mhmd-sh3rawy)

**Abdalla Elmougi**
- GitHub: [@Elmougi](https://github.com/Elmougi)

**Mazen Abdallah**
- GitHub: [Mazen657](https://github.com/Mazen657)

**Ahmed Adel**
- GitHub: [@Ahmedd226](https://github.com/Ahmedd226)

**Ahmed Ayman**
- GitHub: [@Ahmed_Ayman](https://github.com/AhmedAyman12gh)

## 🙏 Acknowledgments

- NTI (National Telecommunication Institute) for project guidance

---

⭐ **Star this repo if you find it helpful!** ⭐