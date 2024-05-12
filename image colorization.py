#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import os
from tkinter import filedialog
from tkinter import Tk
from matplotlib import pyplot as plt



# In[2]:


# Initialize Tkinter
root = Tk()
root.withdraw()  # Hide the main window



# In[3]:


# Paths to load the model
DIR = "C:\\Users\\grees\\OneDrive\\Documents\\proj"
PROTOTXT = os.path.join(DIR, "colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, "pts_in_hull.npy")
MODEL = os.path.join(DIR, "colorization_release_v2.caffemodel")



# In[4]:


# Load the Model
print("Load model")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)



# In[5]:


# Load centers for ab channel quantization used for rebalancing.
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]



# In[6]:


# Tkinter file dialog to get the image path
image_path = filedialog.askopenfilename(title="Select Black and White Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])


# In[7]:


if not image_path:
    print("No image selected. Exiting.")
    root.destroy()
else:
    root.destroy()  # Close the Tkinter window

    


# In[8]:


# Load the input image
image = cv2.imread(image_path)
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    


# In[9]:


resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50



# In[10]:


print("Colorizing the image")
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

colorized = (255 * colorized).astype("uint8")

   


# In[11]:


# Display the original and colorized images in the notebook
plt.figure(figsize=(10, 5))

   


# In[12]:


plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

   


# In[13]:


plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB))
plt.title("Colorized Image")
plt.axis("off")

plt.show()


# In[ ]:




