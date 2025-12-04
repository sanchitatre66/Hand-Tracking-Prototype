# Hand Tracking Proximity Prototype  
### Classical Computer Vision — OpenCV + NumPy (No MediaPipe / No Deep Learning)

This repository contains a **real-time hand tracking prototype** developed as part of the **Arvyax internship assignment**.  
The system tracks the user’s hand using only **classical computer vision techniques**, and classifies the interaction with a **virtual object boundary** into:

- 🟢 **SAFE**  
- 🟡 **WARNING**  
- 🔴 **DANGER**

This project fully follows assignment rules:  
**No MediaPipe, No OpenPose, No Cloud APIs — Only OpenCV + NumPy.**

---

## 🚀 Features

### ✔ Classical Computer Vision (No ML Models)
- HSV skin segmentation with calibration  
- Convex hull extraction for stable hand detection  
- Contour filtering (area, solidity, aspect ratio)  
- Motion detection  
- Face exclusion using Haar Cascade  
- Background subtraction  
- Distance-based interaction logic  

### ✔ Real-Time Performance
- Runs at **8+ FPS** on CPU  
- Lightweight and optimized  

### ✔ Visual Feedback Overlay
- Live webcam feed  
- Convex hull around detected hand  
- Virtual object boundary (white box)  
- Current state (SAFE / WARNING / DANGER)  
- Red **“DANGER DANGER”** alert when hand is too close  

---

## 📦 Project Structure

```
.
├── hand_prototype_convex.py     
├── README.md                   
├── requirements.txt             
             
```

---

## 🧠 How the System Works

### 1️⃣ Skin Segmentation  
Hand region is detected using HSV thresholds.  
User clicks on their **palm** after pressing `c` to calibrate the skin tone.

### 2️⃣ Motion-Based Filtering  
Only moving skin-like regions are kept to reduce false background detection.

### 3️⃣ Face Exclusion  
Detected face region is removed using a Haar Cascade to avoid misclassification.

### 4️⃣ Convex Hull Extraction  
A convex hull is drawn around the largest valid contour for a clean hand shape.

### 5️⃣ Distance-Based State Calculation  
Hand center is compared with the virtual rectangle.

| Distance | State     |
|----------|-----------|
| Far      | SAFE      |
| Near     | WARNING   |
| Very Near / Inside | DANGER |

### 6️⃣ Visual Overlays  
- Current state label  
- Convex hull + hand center  
- Virtual object rectangle  
- "DANGER DANGER" warning  

---

## ▶️ How to Run the Prototype

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Run the Prototype**
```bash
python demo.py
```

---

## 🎮 Controls

| Key | Function |
|-----|----------|
| **c** | Calibrate skin tone (click on your palm) |
| **+** | Increase HSV margin |
| **-** | Decrease HSV margin |
| **m** | Toggle mask-debug view |
| **q / Esc** | Quit program |

---

## 🧪 Calibration Guide

To ensure accurate hand detection:

1. Press **c**  
2. Move mouse pointer to your **palm**  
3. Left-click to capture skin tone  
4. Press **+** if detection is weak  
5. Press **m** to view motion + skin masks  

---

## 🔧 Dependencies

```
opencv-python
numpy
```

Included in `requirements.txt`.

---

## 🙋‍♂️ Author  
**Sanchit Atre**  
Hand Tracking Prototype – Classical Computer Vision  
Python | OpenCV

