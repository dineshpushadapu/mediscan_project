# 👁️ MediScan: AI-Powered Ocular Disease Diagnosis

An enterprise-grade deep learning solution for automated diagnosis of ocular diseases from retinal fundus photographs. **MediScan** leverages a fine-tuned **VGG16 Architecture** built with **TensorFlow/Keras** and an interactive **Streamlit** web dashboard to deliver high-precision classification, real-time image validation, and actionable clinical remedy suggestions.

---

## 🏛️ System Architecture

```
                          +-----------------------------------+
                          |      Retinal Fundus Image         |
                          +-----------------+-----------------+
                                            |
                                            v
                          +-----------------------------------+
                          |   Automated Scan Validation       |
                          |  - Dark Corner & Uniformity Check |
                          |  - Retinal Tissue Spectrum Check  |
                          +-----------------+-----------------+
                                            |
                                   (Valid Fundus Scan)
                                            v
                          +-----------------------------------+
                          |  Image Preprocessing (224x224 RGB)|
                          +-----------------+-----------------+
                                            |
                                            v
                          +-----------------------------------+
                          |    VGG16 Transfer Learning Model  |
                          +-----------------+-----------------+
                                            |
                                            v
                          +-----------------------------------+
                          |  Multi-Class Softmax Predictions  |
                          +-----------------+-----------------+
                                            |
          +-------------------+-------------+-------------+-------------------+
          |                   |                           |                   |
          v                   v                           v                   v
     [ Cataract ]    [ Diabetic Retinopathy ]        [ Glaucoma ]        [ Normal Eye ]
          |                   |                           |                   |
          +-------------------+-------------+-------------+-------------------+
                                            |
                                            v
                          +-----------------------------------+
                          |   Streamlit Interactive Dashboard |
                          | - Confidence Scores & Breakdown   |
                          | - Medical Remedy Guidance         |
                          +-----------------------------------+
```

---

## ✨ Features & Capabilities

- 👁️ **Multi-Class Disease Diagnosis**: Classifies fundus photographs into **4 categories**: Cataract, Diabetic Retinopathy, Glaucoma, and Healthy (Normal).
- 🛡️ **Intelligent Retinal Scan Validation**: Automatically analyzes background darkness uniformity and spectral distribution of the uploaded image to reject non-retinal photos before model inference, preventing false diagnostics.
- 📊 **Detailed Probability Breakdown**: Displays exact confidence metrics and progress bars for all 4 condition classes.
- 💊 **Actionable Medical Guidance**: Provides disease-specific medical remedy suggestions (surgeries, eye drops, anti-VEGF injections, lifestyle management).
- 🎨 **Modern Responsive Interface**: Styled Streamlit dashboard with sidebar diagnostics info, image preview, and clean status indicators.
- ☁️ **Cloud & Container Ready**: Includes pre-configured `render.yaml` and `Procfile` for seamless one-click deployment to hosting platforms like Render or Heroku.

---

## 🩺 Supported Ocular Conditions

| Condition | Description | Clinical Remedy / Action |
| :--- | :--- | :--- |
| **Cataract** | Clouding of the crystalline lens impairing vision | Lens replacement surgery (Intraocular Lens / IOL insertion). |
| **Diabetic Retinopathy** | Damage to retinal blood vessels caused by diabetes | Anti-VEGF injections, laser photocoagulation, strict glucose control. |
| **Glaucoma** | Optic nerve damage caused by abnormally high intraocular pressure | Intraocular pressure-lowering eye drops, laser trabeculoplasty, surgery. |
| **Normal** | Healthy retinal anatomy with no detectable pathology | Regular annual eye exams and routine ocular care. |

---

## 📂 Project Structure

```
medi_scan_project/
├── app.py                     # Entrypoint wrapper for Streamlit interface
├── requirements.txt           # Python package dependencies
├── render.yaml                # Render platform deployment configuration
├── Procfile                   # Process file for Heroku/Render deployment
├── .gitignore                 # Excluded files (virtualenvs, weights, caches)
├── model.h5                   # Fine-tuned VGG16 TensorFlow model weights (local)
├── Medi_Scan.ipynb            # Jupyter notebook for model training & evaluation
├── User_interface/
│   ├── main.py                # Main Streamlit web application & validation logic
│   └── eyejpg.jpg             # Sidebar logo asset
├── Dataset/                   # Retinal fundus image datasets
└── test_images/               # Sample retinal images for testing
```

---

## ⚙️ Installation & Local Setup

### Prerequisites
- **Python 3.10+** installed on your system.
- Git for repository management.

### 1. Clone the Repository
```bash
git clone https://github.com/dineshpushadapu/mediscan_project.git
cd mediscan_project
```

### 2. Create & Activate Virtual Environment
- **Windows**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Project

Ensure the trained model file `model.h5` is placed in the project root directory, then run:

```bash
streamlit run app.py
```

The web dashboard will automatically launch in your browser at `http://localhost:8501`.

---

## 🧪 Model Architecture & Training

- **Base Architecture**: VGG16 (ImageNet pre-trained weights)
- **Input Resolution**: `224 x 224 x 3` (RGB)
- **Feature Extractor**: Convolutional blocks with Max Pooling
- **Classification Head**: Dense layers with Dropout regularization and 4-unit Softmax output.
- **Optimization**: Adam optimizer with Categorical Crossentropy loss.

---

## ☁️ Deployment

### Render Deployment
This project includes a ready-to-use `render.yaml` configuration:
1. Connect your GitHub repository to [Render](https://render.com).
2. Create a new **Web Service**.
3. Render will auto-detect `render.yaml` and run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```

---

## ⚠️ Disclaimer

> [!WARNING]
> **MediScan** is designed for educational, research, and diagnostic assistance purposes only. It should not be used as a sole diagnostic tool for clinical decision-making. Always consult a licensed ophthalmologist or healthcare professional for professional diagnosis and medical advice.

---

## 🤝 Author & Acknowledgments

- **Developer**: Dinesh Samba Siva Rao Pushadapu ([@dineshpushadapu](https://github.com/dineshpushadapu))
- **Built With**: TensorFlow, Keras, Streamlit, OpenCV, Pillow, PyData Stack.
