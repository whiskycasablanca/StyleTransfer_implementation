# 🎨 Neural Style Transfer Implementation

This repository implements **Neural Style Transfer**, based on **"A Neural Algorithm of Artistic Style" (Gatys et al., 2015)**. The model applies deep learning techniques to blend the artistic style of one image with the content of another.

## 📌 Features

- Implements **Adam** and **L-BFGS** optimizers for training
- Compares different **initialization strategies** for L-BFGS (random vs. content image)
- Uses **VGG-19** as the feature extractor
- Custom loss functions (`loss.py`), model definitions (`models.py`), and training script (`train.py`)
- 

## 📂 Repository Structure

📦 StyleTransfer_Implementation

┣ 📜 [README.md](http://readme.md/)                # Project documentation

┣ 📜 content_disastergirl.jpg # Content image

┣ 📜 style_hockney.jpg        # Style image

┣ 📜 [loss.py](http://loss.py/)                  # Defines loss functions (content loss, style loss, etc.)

┣ 📜 [models.py](http://models.py/)                # Defines the neural network architecture (VGG-19 based)

┣ 📜 [train.py](http://train.py/)                 # Training script for style transfer


## 🖼️ Style Transfer Results

Below are the final results of the Adam and L-BFGS optimizers with different initialization strategies. 

For the content image, I used the famous meme Disaster Girl(2005), and for the style image, I chose Portrait of an Artist(1973) by David Hockney.

![Disaster Girl](/content_disaster_girl.jpg)

![Portrait of an Artist](/style_hockney.jpg)  


### 🔹 Adam Optimizer (Learning Rate: 1e-1)

![](results/result_adam.jpg)

Final Loss: **45.67**

---

### 🔹 L-BFGS Optimizer (Content Image Initialization, Learning Rate: 1)

![](results/result_lbfgs_content.jpg)

Final Loss: **28.34**

---

### 🔹 L-BFGS Optimizer (Random Initialization, Learning Rate: 1)

![](results/result_lbfgs_random.jpg)

Final Loss: **30.12**

---

### 🔬 **Observations**

- **L-BFGS converges faster** than Adam, achieving a lower final loss.
- **Content image initialization for L-BFGS** results in a **more stable** and structured stylization.
- **Random initialization with L-BFGS** sometimes produces better artistic effects but can be **unstable**.
