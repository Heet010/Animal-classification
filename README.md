# Cat vs. Dog Classification using Custom ResNet in PyTorch  

This project is a binary classification task to distinguish between cats and dogs using a custom ResNet architecture implemented in PyTorch. No pre-trained models or transfer learning techniques are used.  

---

## Table of Contents  
1. [Dataset Description](#dataset-description)  
2. [Preprocessing Pipeline](#preprocessing-pipeline)  
3. [Model Design](#model-design)  
4. [Training and Evaluation](#training-and-evaluation)  
5. [Model Testing & Prediction](#model-testing--prediction)  
6. [Results](#resuls)  
7. [Challenges and Solutions](#challenges-and-solutions)   

---

## Dataset Description  
This project utilizes the [Animal Dataset](https://www.kaggle.com/datasets/arifmia/animal/data) from Kaggle. Only the "cats" and "dogs" subfolders are used for binary classification, while the "horses" subfolder is ignored.  

- **Classes**: Cat, Dog  
- **Image Format**: JPG  
- **Image Size**: Variable 
- **Color Mode**: RGB 

---

## Preprocessing Pipeline  
To ensure consistency and optimal input for the ResNet model, the following preprocessing steps are applied:  
- **Image Resizing**: 
    All images are resized to 64x64 pixels.  
- **Grayscale Conversion**: 
    Images are converted to grayscale to reduce complexity.
- **Random Cropping with Padding**: 
    Randomly crop the images to a size of 64x64 with padding of 4 pixels. This helps with image augmentation and model robustness. 
- **Color Jitter**:  
    Randomly adjust the brightness, contrast, saturation, and hue of the images within defined ranges to simulate various lighting conditions.    
- **Normalization**:  
   Normalize the pixel values of the images to have a mean of 0.5 and standard deviation of 0.5. This step helps in improving model convergence.    

The preprocessing pipeline is implemented using a PyTorch `DataLoader`, ensuring that images are processed online during training.  

---

## Model Design  
This project utilizes a custom version of the ResNet architecture. Key modifications include:  
- **Input Layer**: Adjusted to accept 1-channel grayscale images (instead of 3) of size 64x64 .  
- **Output Layer**: Modified to output 2 classes (Cat and Dog).  
- **Intermediate Layers**: Maintains the overall ResNet structure but added maxpooling layer after first convolution layer to reduce training time and changed kernel size to better suit the particular layer (menrioned with the help of comments in model).   

---

## Training and Evaluation  
- **Loss Function**: CrossEntropyLoss.  
- **Optimizer**: Adam optimizer with learning rate 0.0001.  
- **Metrics**: Max validation accuracy = 99.7 % and max F1-Score = 0.9953 has been achieved during 17th and 20th epoch of training.  
- **Model Checkpointing**: Automatically saves the best model based on validation accuracy please check check point folder.  

### Visualization:  
- Different curves are ploted for visulization like validation loss, training loss, validation accuracy curve and validation f1 score.  

---

## Model Testing & Prediction 

A dedicated prediction script (test.py) is provided that:  
- Accepts an input image and a model checkpoint path.  
- Outputs the predicted class ("Cat" or "Dog") and a confidence score. 

## Results
Model has been tested on testset an following are the best results which has been achieved:
- Test Accuracy: 0.9968, 
- Test F1 Score: 0.9952,
- Precision: 0.9925, 
- Recall: 0.9980 

## Challenges and solution 
Mostly challenges occured in model design and as a solution necessary changes has been done in model like adjust the color channel, output class and kernel size dring processing the intermediate layers (mentioned with the help of comments in model).
  

