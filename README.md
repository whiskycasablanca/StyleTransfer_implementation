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



## ✏️ Implementation Details

### models.py
This file contains the main architecture for extracting style and content features using a pre-trained VGG19 model. Below is a brief overview:

- **Imports and Layer Map**  
  `torch` and `torch.nn` are used for PyTorch-based operations and module definitions. The `vgg19` from `torchvision.models` provides the pretrained VGG19 network. The `conv` dictionary maps layer names (e.g., `conv1_1`, `conv4_2`) to indices in the VGG19 feature extractor.

- **StyleTransfer Class**  
  Inherits from `nn.Module`, initializes a pre-trained VGG19 model, and stores the indices of layers required for style and content extraction.

- **forward Method**  
  Receives an input tensor `x` and a mode (`"style"` or `"content"`), passes `x` through each layer, and collects style or content features for later use in the style transfer process.

---

### loss.py
This file defines the loss functions used to measure how closely the generated image matches the content and style targets. Below is a brief overview:

- **ContentLoss Class**  
  Calculates the mean squared error (`F.mse_loss`) between the feature representations of the generated image and the content image. This ensures the generated image retains the structural integrity of the content.

- **StyleLoss Class**  
  Uses a Gram matrix representation to compute style similarity. The `gram_matrix` function transforms feature maps, and the loss is the mean squared error between these Gram matrices of the generated and style images.

---

### train.py
This file orchestrates the entire style transfer process, from loading images to running the optimization loop. Below is a brief overview:


- **Data Preprocessing and Postprocessing**  
  - **`pre_processing(image)`** resizes and normalizes the input image, converting it to a Tensor ready for the model.  
  - **`post_processing(tensor)`** converts the output Tensor back to a PIL image, reversing normalization and adjusting pixel values.

- **Training Routine**  
  - **`train_main()`** loads content and style images, sets up the `StyleTransfer` model, and initializes the optimizer (e.g., `LBFGS`).  
  - A closure function calculates the total loss (content + style), backpropagates, and updates the generated image `x`.  
  - Saves intermediate outputs periodically and prints the current loss values.

- **Execution**  
  When the file is run directly (`if __name__ == "__main__":`), the `train_main()` function is called to perform style transfer and save the results.


## 🖼️ Style Transfer Results

Below are the final results of the Adam and L-BFGS optimizers with different initialization strategies. 

For the content image, I used the famous meme Disaster Girl(2005), and for the style image, I chose Portrait of an Artist(1973) by David Hockney.

## 🖼️ Content & Style Images  
Below are the images used for style transfer.  

<table>
  <tr>
    <td><b>Content Image (Disaster Girl)</b></td>
    <td><b>Style Image (Portrait of an Artist)</b></td>
  </tr>
  <tr>
    <td><img src="content_disastergirl.jpg" width="400"></td>
    <td><img src="style_hockney.jpg" width="400"></td>
  </tr>
</table>

### 🔹 Adam Optimizer ((Random Initialization, Learning Rate: 1e-1, epoch 900)

<img src="results/DisasterGirl_Adam900_R-Init.png" width="350">

Content Loss: **2.82**
Style Loss: **3.21**
Total Loss: **6.03**


### 🔹 Adam Optimizer (Content Image Initialization, Learning Rate: 1e-1, epoch 900)


<img src="results/DisasterGirl_Adam900_C-Init.png" width="350">
  

Content Loss: **1.51**
Style Loss: **1.85**
Total Loss: **3.36**

---

### 🔹 L-BFGS Optimizer (Content Image Initialization, Learning Rate: 1, epoch 200)


<img src="results/DisasterGirl_LBFGS200_C-Init.png" width="350">

Content Loss: **1.30**
Style Loss: **0.88**
Total Loss: **2.18**

---

### 🔹 L-BFGS Optimizer (Random Initialization, Learning Rate: 1, epoch 200)

<img src="results/DisasterGirl_LBFGS200_R-Init.png" width="350">

Content Loss: **1.43**
Style Loss: **0.86**
Total Loss: **2.34**


---

## 🔬 **Observations**

- **L-BFGS converges faster** than Adam, achieving a lower final loss.
- **Content image initialization for L-BFGS** results in a **more stable** and structured stylization.
- **Random initialization with L-BFGS** sometimes produces better artistic effects but can be **unstable**.
- **Using the content image for initialization significantly speeds up convergence**, reducing the number of optimization steps needed.
- **Content image initialization also leads to a lower final loss**, indicating that the optimization process finds a better solution compared to random initialization.


## 🔗 **References**
- **Paper:** *A Neural Algorithm of Artistic Style*, Gatys et al., 2015  
- **Lecture:** *Neural Style Transfer Implementation*, [AI Research Engineer를 위한 논문 구현 시작하기 with PyTorch] by [화이트박스]  
