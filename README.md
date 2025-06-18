# 📷 PXR - 2D to 3D with AR

**PXR** is a mobile application built with **Flutter** and a **Python FastAPI** backend. It allows users to upload 2D images and get a 3D reconstruction, which can be viewed in Augmented Reality. The 3D generation is powered by a PyTorch model on the backend.

---

## 🚀 Features

### 📱 Mobile App
- Upload 2D images (camera/gallery)
- View 3D results in a model screen
- Firebase Auth for login/signup

### 🧠 ML Backend
- FastAPI server running a PyTorch model
- Accepts image uploads and returns voxel predictions

---

## 🛠️ Tech Stack

- **Frontend:** Flutter
- **Backend:** FastAPI (Python)
- **ML:** PyTorch
- **Database:** Firebase Firestore
- **Authentication:** Firebase Auth

---



---

## ⚙️ Setup Instructions

### ✅ Clone the Repo

```bash
git clone https://github.com/Slayer9966/PXR.git
cd PXR
```

---

### 📱 Flutter Frontend Setup

```bash
flutter pub get
flutter run
```

Make sure you’re logged into Firebase and your emulator/device is connected.

---

### 🧠 Backend Setup (FastAPI + ML)

```bash
cd Server
python -m venv venv
venv\Scripts\activate       # For Windows
# or
source venv/bin/activate    # For macOS/Linux

pip install -r requirements.txt
```

---

### 📥 Download Model Weights

> ⚠️ GitHub doesn't support files larger than 100MB, so you must download the model weights manually.

[📦 Download fixed_best_model.pth](https://drive.google.com/file/d/1U1Hr8hPXtdea3P1hwpm2UL_A7efxze_T/view?usp=sharing)

Place the downloaded file in this location:

```bash
Server/ml_models/fixed_best_model.pth
```

---

### 🚀 Run FastAPI Server

```bash
uvicorn main:app --reload
```

This will start the backend server at:



---

## 🔐 Firebase Setup

- Make sure `google-services.json` is added in `android/app/`
- Ensure Firebase Auth and Firestore are configured in the Firebase Console

---

## 📌 Notes

- This project is for academic/demo use only
- Weights are not included in the repo (due to size limit)
- Model predictions return voxel data for a 3D object

---

## 📜 License

Licensed under the [MIT License](LICENSE) — use, modify, and distribute freely.

---

## 🙋‍♂️ Author

**Syed Muhammad Faizan Ali**  
📍 Islamabad, Pakistan  
📧 faizandev666@gmail.com  
🔗 [GitHub](https://github.com/Slayer9966) | [LinkedIn](https://www.linkedin.com/in/faizan-ali-7b4275297/)
