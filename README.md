# 🧠 Advanced Stroke Prediction Dashboard  

An interactive **Machine Learning & Data Visualization web app** built with **Streamlit** to analyze and predict stroke risk from **multimodal medical image data**.  
This project integrates computer vision and ML models to extract meaningful insights and enable stroke risk interpretation using a modern web-based dashboard.

---

## 🚀 Project Overview  

This project aims to leverage **multimodal image data (CT & MRI scans)** and associated patient features to **predict stroke types and analyze patterns**.  
The app offers **AI-powered analysis**, **interactive visualizations**, and a **Streamlit dashboard** for medical insights and research use.

---

## ✨ Key Highlights  

- 🧩 Combines **image-based and tabular data** for multimodal stroke analysis  
- 📊 **Interactive charts and visual analytics** for exploring patterns  
- 🤖 **AI model integration** for classification of stroke types  
- 🧠 **Deep learning & traditional ML** hybrid pipeline  
- 🎨 Elegant Streamlit UI for smooth user interaction  
- ☁️ Ready for cloud deployment via Streamlit Cloud  

---

## 📊 Dataset Description  

**Dataset Source:** [Multimodal Stroke Image Dataset (Kaggle)](https://www.kaggle.com/datasets/turkertuncer/multimodal-stroke-image-dataset?select=deep)

The dataset contains **CT and MRI stroke images** with corresponding labels to identify stroke conditions.  

### 🩻 Data Overview  

| Data Type | Description |
|------------|-------------|
| CT Images | Computed Tomography scans representing various stroke types |
| MRI Images | Magnetic Resonance Imaging scans with stroke localization |
| Labels | Stroke category classification labels (e.g., ischemic, hemorrhagic, normal) |
| Directory Structure | Images are organized in folders according to stroke class |
| Usage | Ideal for training CNNs or multimodal deep learning models |

**Note:** Dataset contains multiple modalities and subfolders under `/deep` for each stroke type.  

---

## 🧠 Features  

- 🩺 **Stroke Type Classification:** Predict stroke type using ML/DL models  
- 📈 **Visual Data Analysis:** Explore dataset statistics, image distributions, and class balance  
- 🔍 **Interactive Image Viewer:** View CT/MRI scans dynamically  
- 🧮 **Hybrid Pipeline:** Combines CNN feature extraction + ML classification  
- 📊 **Statistical Insights:** Identify relationships among stroke parameters  
- 🌈 **Custom Streamlit UI:** Clean, modular, and responsive interface  

---
## 📊 Tech Stack
| Category          | Tools / Libraries           |
| ----------------- | --------------------------- |
| Web Framework     | Streamlit                   |
| Image Processing  | OpenCV, Pillow              |
| Data Analysis     | Pandas, NumPy               |
| Visualization     | Plotly, Matplotlib, Seaborn |
| Machine Learning  | Scikit-learn                |
| Deep Learning     | TensorFlow / Keras          |
| Model Persistence | joblib / pickle             |
| Environment       | Python 3.8+                 |

