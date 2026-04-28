# 🎨 Neural Style Transfer App

A deep learning application that transforms ordinary images into artistic styles using Neural Style Transfer.
Built with **PyTorch** and **Streamlit**, this project demonstrates both **real-time inference** and **optimization-based style transfer** techniques.

---

## 🔗 Live Demo

👉 https://neural-style-transfer-app-hkazyhuftibcmwqxgujnog.streamlit.app/

---

## 🚀 Features

* Upload any image and apply artistic styles instantly
* Multiple pre-trained styles:

  * Candy
  * Mosaic
  * Rain Princess
  * Udnie
* Fast style transfer using feed-forward neural networks
* Optimization-based style transfer using VGG19
* Download stylized output
* Clean and interactive Streamlit UI

---

## 🧠 Core Concepts

This project implements **two major approaches**:

### 1. ⚡ Fast Style Transfer (Feed-Forward Network)

* Uses a trained **Transformer Network**
* Performs style transfer in a **single forward pass**
* Extremely fast (real-time)

### 2. 🧪 Optimization-Based Style Transfer

* Uses **VGG19** as a feature extractor
* Minimizes:

  * **Content Loss** → preserves structure
  * **Style Loss** → captures artistic texture using Gram Matrix
* Uses **LBFGS optimizer** for iterative refinement

---
## 🧪 Technical Highlights

* Implemented Gram Matrix computation from scratch for style representation
* Tuned content vs style weight trade-off for perceptual quality
* Optimized inference speed using cached model loading in Streamlit
* Handled model size constraints for cloud deployment

---

## 🏗️ Tech Stack

* Python
* PyTorch
* Torchvision
* Streamlit
* Pillow (PIL)
* NumPy

---

## 📁 Project Structure

```
neural-transfer-project/
│
├── app.py                  # Streamlit UI
├── style.py                # Style transfer logic
├── transformer_net.py      # Fast style model architecture
├── utils.py                # Image utilities
├── vgg.py                  # VGG feature extractor
│
├── images/
│   └── style-images/       # Style reference images
│       ├── candy.jpg
│       ├── mosaic.jpg
│       ├── rain_princess.jpg
│       └── udnie.jpg
│
├── saved_models/           # Pre-trained models (.pth)
│       ├── candy.pth
│       ├── mosaic.pth
│       ├── rain_princess.pth
│       └── udnie.pth
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/neural-style-transfer-app.git
cd neural-style-transfer-app

pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📸 Example Output

Transforms a normal image into an artistic painting style using deep neural networks.

---

## ⚡ Key Engineering Challenges

* Cross-platform path issues (Windows vs Linux)
* Managing large `.pth` model files
* Handling image preprocessing and normalization
* Debugging deployment errors on Streamlit Cloud
* Efficient model loading and caching

---

## 💡 Key Learnings

* Practical use of **CNN feature extraction (VGG19)**
* Understanding **Gram Matrix for style representation**
* Trade-offs between **speed vs quality** in ML systems
* Importance of **file structure and deployment debugging**
* Building end-to-end ML applications (not just models)

---

## 🚀 Future Improvements

* Add more artistic styles dynamically
* GPU acceleration support
* Before/After slider comparison UI
* Video style transfer
* Convert into REST API + frontend
* Deploy on scalable cloud platforms (AWS/GCP)

---

## 👤 Author

**Bhavay Vij**

---

## ⭐ If you found this useful

Give it a star ⭐ and try different styles!
