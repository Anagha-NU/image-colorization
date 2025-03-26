Black & White Image Colorization using OpenCV and Deep Learning
📌 Overview
This project colorizes black-and-white images using a deep learning model based on OpenCV's Deep Neural Network (DNN) module. The model is pre-trained and uses Caffe for colorization.

⚡ Features
Uses a deep learning model to colorize grayscale images.

Supports user-friendly image selection via Tkinter file dialog.

Utilizes OpenCV's DNN module for fast and efficient processing.

Displays the original and colorized images using Matplotlib.

🛠️ Technologies Used
Python

OpenCV

NumPy

Matplotlib

Tkinter



📸 How It Works
The program prompts you to select a black-and-white image using a file dialog.

It loads the image and preprocesses it by converting it into LAB color space.

The deep learning model predicts the A and B color channels.

The predicted colorized image is displayed alongside the original.
