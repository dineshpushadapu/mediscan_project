
# 👁️ Medi-Scan: AI-Powered Ocular Disease Diagnosis

Medi-Scan is a deep learning-based web application designed to classify ocular diseases including **cataract**, **glaucoma**, **diabetic retinopathy**, and **normal** retina. It uses a trained CNN model and provides predictions through a user-friendly Streamlit interface.

---

## 📂 Project Structure

```
MEDI_SCAN_PROJECT/
│
├── Dataset/
│   └── linkofdataset.txt         # Text file with link to the dataset used
│
├── test_images/                  # Sample test images categorized by disease
│   ├── cataract/
│   ├── diabetic_retinopathy/
│   ├── glaucoma/
│   └── normal/
│
├── User_interface/
│   └── eyejpg.jpg                # UI background or image asset
│
├── model.h5                      # Trained CNN model file
├── Medi_Scan.ipynb               # Jupyter Notebook with training pipeline
├── main.py                       # Streamlit-based frontend app
├── requirements.txt              # List of dependencies
└── .gitignore                    # Files/folders to ignore in version control
```

---

## 🔧 Technologies Used

- **Python 3.9+**
- **TensorFlow/Keras** – For building and training the CNN
- **OpenCV, NumPy, Pandas** – Image processing and data handling
- **Streamlit** – Web interface for uploading and classifying images

---

## 🧠 Key Features

- Classifies retinal images into: Cataract, Glaucoma, Diabetic Retinopathy, or Normal.
- Model trained on curated retinal image datasets with >95% accuracy.
- Real-time prediction through an intuitive web interface.
- Supports drag-and-drop image upload.

---

## 🛠️ Steps to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/MEDI_SCAN_PROJECT.git
cd MEDI_SCAN_PROJECT
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```bash
streamlit run main.py
```

### 5. Use the Interface
- Upload a retinal image via the Streamlit UI
- The app will classify and display the result along with the confidence level

---

## 🔗 Dataset

The dataset used is linked inside:
```
Dataset/linkofdataset.txt
```
Please download the data and structure it as shown under `test_images/`.

---

## 📸 Sample Output

- Input: `glaucoma_image.jpg`
- Output: **Prediction: Glaucoma (Confidence: 96.2%)**

---

## 📥 Download This Project

👉 [Click here to download the full project as ZIP](https://example.com/medi-scan-download)  
_Replace this with your actual GitHub or Drive download link._

---

## 📜 License

This project is open-source and free for research and academic use.  
© 2025 P. Dinesh Samba Siva Rao
