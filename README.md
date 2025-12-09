# 🖐️ Hand Safety Distance Monitoring System

This project is a **hand-distance monitoring application** built using Python and OpenCV.
It detects the user's hand using a pre-trained hand tracking model, expands the palm ROI to include fingers, identifies the closest point on the hand contour, and computes the distance to a defined safety-zone rectangle in the top-right corner of the screen.

If the hand gets too close to this danger zone, the system displays a **warning message** in real time.

---

## 🚀 Features

### ✔ **Palm & Hand Detection**

Hand detection is powered by the pretrained model from
👉 **[https://github.com/victordibia/handtracking](https://github.com/victordibia/handtracking)**

### ✔ **ROI Expansion**

Palm bounding box is expanded safely within image boundaries to include fingers.

### ✔ **Contour-Based Hand Tracking**

The system extracts the maximum hand contour and computes the nearest point to the target rectangle.

### ✔ **Distance Calculation**

The shortest distance between:

* the **top-right rectangle zone**, and
* the **closest point on the detected hand**

is calculated using `cv2.pointPolygonTest()` and contour geometry.

### ✔ **Warning System**

Displays a live warning if the hand enters a near-danger region.

### ✔ **Gesture-Friendly**

Designed to support further gesture-based interactions.

---

## 🧰 Tech Stack

* **Python**
* **OpenCV**
* **TensorFlow**
* **MobileNetV2-SSD Hand Tracking model**
* Contour Geometry & Computer Vision Techniques

---

## 📦 Installation

### 1️⃣ Install Dependencies

```bash
pip install opencv-python tensorflow numpy
```

Make sure you have **TensorFlow 2.x** installed (GPU optional).

### 2️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/hand-safety-distance
cd hand-safety-distance
```

### 3️⃣ Download Hand Tracking Model

This project uses the pretrained MobileNetV2-SSD model from:
👉 [https://github.com/victordibia/handtracking/](https://github.com/victordibia/handtracking/)

Download the model folder (`models/`) and place it inside this project directory.

---

## ▶️ Running the Application

```bash
python app.py
```

Your webcam feed will open, and:

* A fixed rectangle appears in the top-right corner.
* Your hand is detected inside the frame.
* App measures how close your hand gets to the rectangle.
* A warning message is displayed if the distance falls below the threshold.

---

## 📁 Project Structure (Suggested)

```
hand-safety-distance/
│── models/                # Pretrained hand detection model
│── app.py                 # Main application script
│── utils.py               # Helper functions (ROI expansion, distance)
│── README.md
│── requirements.txt
```

---

## 🙏 Acknowledgement

This project **cites and uses** the excellent work from:

### 👉 [https://github.com/victordibia/handtracking/](https://github.com/victordibia/handtracking/)

Full credit to the original author for the pretrained MobileNetV2-SSD hand tracking model.

---


