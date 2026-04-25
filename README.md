# 🎨 Neural Style Transfer App

Transform your photos into artistic masterpieces using Deep Learning.

This project implements **Neural Style Transfer (NST)** using a pre-trained **VGG19 Convolutional Neural Network**, allowing users to blend the *content* of one image with the *style* of another.

---

## 🚀 Features

* 🧠 Deep Learning based style transfer (VGG19)
* 🎚 Adjustable style strength (real-time control)
* 🖼 Side-by-side comparison (Original vs Stylized)
* ⚡ Interactive web app using Streamlit
* 📥 Download generated images
* 🎨 Works with any content and style images

---

## 🧩 How It Works

The model separates and recombines:

* **Content** → structure of the image
* **Style** → textures, colors, patterns

It uses:

* Content Loss
* Style Loss (Gram Matrix)
* Optimization using LBFGS

---

## 🛠 Tech Stack

* Python
* PyTorch
* TorchVision
* Streamlit
* PIL (Image Processing)

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open in browser:

http://localhost:8501

---

## 📸 Demo

### 🔹 Input Images
![UI](<images ui.png>)

Upload:

* Content Image
![alt text](image.png)
* Style Image
![alt text](image-1.png)

### 🔹 Output
![alt text](<Screenshot 2026-04-25 212937.png>)
* Stylized Image with adjustable intensity

👉 **(Add your screenshots here after running the app)**

Example:

![UI Screenshot](images/ui.png)
![Output Screenshot](outputs/output.png)

---

## 📁 Project Structure

```
neural-style-transfer-app/
│
├── app.py              # Streamlit UI
├── model.py            # Style Transfer logic
├── requirements.txt
├── README.md
│
├── images/             # Input images
├── outputs/            # Generated outputs
├── notebook/           # Experiment notebook
```

---

## 🎯 Key Learning Outcomes

* Understanding CNN feature extraction
* Difference between content and style representation
* Gram Matrix for style computation
* Optimization-based image generation
* Building interactive ML applications

---

## ⚠️ Limitations

* Slow (optimization-based approach)
* Not real-time
* CPU execution takes time
* Output quality depends on input images

---

## 🚀 Future Improvements

* Fast Style Transfer (real-time)
* GPU acceleration
* Deploy online (Streamlit Cloud / HuggingFace)

---

## 🙌 Author

This project was built to explore Deep Learning concepts and practical AI applications using Neural Style Transfer.

---

⭐ If you found this project useful, consider giving it a star!
