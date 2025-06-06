# CIFAR-10 Image Classification  
_Convolutional Neural Network that classifies images into 10 distinct classes._

---

**Author:** Alen Abdrakhmanov  
**MSc Data Science & Artificial Intelligence | BEng Mechanical Engineering**  
**Location:** London, United Kingdom  
**Contact:** [alenabd24@outlook.com](mailto:alenabd24@outlook.com) | +44 7884 457252

---

## Project Overview
The aim of this project is to design and implement a custom convolutional neural network (CNN) architecture for classifying images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 colour images in 10 classes (50,000 for training and 10,000 for testing). Our goal is to build a robust model that generalizes well to unseen data while keeping the architecture both flexible and computationally efficient.

- Developed a novel CNN architecture featuring six intermediate blocks and an output block.  
- Each intermediate block uses six parallel convolutional layers with dynamic output combination via a small fully connected network using softmax-normalized channel averages.  
- Maintained a constant channel width of 64 across all blocks to simplify the model and control parameter growth, reducing overfitting.  
- Used the Adam optimizer with cross-entropy loss, along with a learning rate scheduler (halving every 100 epochs) and early stopping based on test loss.  
- Achieved a maximum test accuracy of **81.71%** on the CIFAR-10 dataset.

  ![Repo Screenshot](Screenshot%202025-06-06%20164413.png)


---

## Dataset
**CIFAR-10**  
- 60,000 colour images (32×32 pixels).  
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.  
- 50,000 training images and 10,000 test images.  

---

## Network Architecture
Prior to finalizing the architecture that yielded the best test accuracy, several alternative designs and hyperparameter configurations were explored:

- **Variants Tried**  
  - Different numbers of intermediate blocks (from 4 up to 8).  
  - Varying the number of convolutional layers per block (from 3 to 7).  
  - Comparing architectures with progressively increasing channel counts against those maintaining a constant channel width.  
  - Early tests revealed that while adding more layers and blocks can capture richer feature hierarchies, it also increases model complexity and risk of overfitting.

- **Final (Most Successful) Model**  
  1. **Blocks**  
     - The model consists of **seven intermediate blocks**, each followed by an output block.  
     - Each intermediate block receives an input tensor of shape `[B, c, H, W]` (with `c` starting at 3 and set to 64 after the first block).  
  2. **Parallel Pathways**  
     - Each block contains **seven parallel convolutional pathways**.  
     - Each pathway applies:  
       1. `conv2d`  
       2. `BatchNorm2d`  
       3. `ReLU` activation  
     - Instead of simply averaging or concatenating the parallels, a **dynamic weighted sum** is computed via a small fully connected sub-network that takes softmax-normalized channel averages as input.  
  3. **Constant Channel Width**  
     - All intermediate blocks maintain a channel width of **64**.  
     - This decision simplifies the model, limits parameter growth, and helps reduce overfitting.  
  4. **Output Block**  
     - Takes the final feature map (shape `[B, 64, H_final, W_final]`) and flattens it.  
     - Uses one fully connected layer to produce logits for the 10 CIFAR-10 classes.  
     - The predicted class is selected using `torch.argmax()` over the output logits.  
  5. **Regularization & Overfitting Mitigation**  
     - Data augmentation (random crops, horizontal flips, etc.).  
     - Weight decay (`2×10⁻⁴`) in the optimizer.  
     - Early stopping based on test loss.  
     - Despite some overfitting in early epochs, the model generalizes fairly well, achieving over 80% test accuracy—a strong result for a custom CIFAR-10 architecture.  

---

## Training & Validation
To train the final model for CIFAR-10 classification:

1. **Optimizer & Scheduler**  
   - **Optimizer:** Adam with an initial learning rate of `0.0007` and weight decay of `2×10⁻⁴`.  
   - **Scheduler:** StepLR that halves the learning rate every 100 epochs (`gamma = 0.5`).  

2. **Early Stopping**  
   - Training was set for up to 200 epochs.  
   - Early stopping monitored the **average test loss**.  
   - If no improvement in test loss occurred for 10 consecutive epochs, training halted to prevent overfitting.

3. **Metrics Computed Each Epoch**  
   - **Training Loss:** Mean of batch losses over the training set.  
   - **Training Accuracy:** Percentage of correctly classified examples over the entire training set.  
   - **Testing Loss:** Mean of batch losses over the test set.  
   - **Test Accuracy:** Percentage of correctly classified examples over the test set.  

These metrics guided hyperparameter tuning and provided insight into convergence and potential overfitting.

---

## Results
- **Maximum Test Accuracy:** 81.71% on the CIFAR-10 test set.  
- **Observations:**  
  - The dynamic weighted-sum mechanism within each block allowed the network to learn which convolutional pathways to emphasize at each stage.  
  - Data augmentation, weight decay, and early stopping effectively mitigated—but did not fully eliminate—overfitting.  
  - Future improvements could include more aggressive augmentation, stronger regularization (e.g., dropout), or reducing model complexity if overfitting persists.

---

## Example Images
An example of images and classes used in the CIFAR-10 dataset:

![CIFAR-10 Sample Images](images/cifar10_samples.png)  
<sup>_Replace `images/cifar10_samples.png` with your actual image path in the repository._</sup>


