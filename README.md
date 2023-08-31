# Brain Tumor Segementation using U-Net and Flask
Get the dataset [here.](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)

## Introduction
The project employs deep learning for the semantic segmentation of MRI scans, specifically targeting tumor regions. 
The goal is to precisely identify and separate tumor areas within the images, contributing to medical diagnosis.
Developed a straightforward website to serve as a user interface.
## Process
1. **Preprocessing:**
   - The dataset is provided in nibabel (.nii) format.
   - Images were resized, and less informative slices of scans were discarded.
   - The dataset was transformed into TFrecords to facilitate efficient preprocessing during each training epoch.
2. **Model:**
   - Employed the U-Net architecture, a deep convolutional neural network, for the task.
   - Its distinctive U-shaped structure includes an encoder to capture context and a decoder for precise localization and residual connections keep track of the original image during decoding.
   - U-Net has been widely adopted in medical imaging due to its effectiveness in segmenting structures of interest from noisy or complex backgrounds.
3. **Training:**
    - Utilized a dataloader with the dataset structured as a tf.data.Dataset.
    - Trained the model for 30 epochs, implementing early stopping with a patience of 5 epochs.
    - Implemented checkpointing to save the optimal weights and accommodate potential failures. Utilized a Kaggle kernel for execution.
4. **Validation:**
    - Attained a cross-entropy loss of 0.02 and an accuracy of 99.01% on the validation dataset.
    - Achieved mean Intersection over Union (IoU) of 0.4 and Dice coefficient of 0.5.
    - refer [this](https://github.com/VeerendraKocherla/Brain-Tumor-Segmentation/blob/main/predictions.ipynb) for predictions made by the model.
5. **WebApp:**
   - Developed a straightforward web application using Flask.
   - Users input relative paths of flair and t1ce images and select the desired tumor area for evaluation.
   - Upon clicking "Generate," the model's prediction for the chosen region is displayed.
## Future work:
  - Train the dataset using a 3D U-Net model that employs conv3d layers instead of conv2d. This approach enables simultaneous extraction of information from all slices, enhancing the model's understanding of the data.
  - Enhance the interactivity of the web app.
  - Explore alternative algorithms such as Autoencoders, GANs, and other methods for experimentation.
## References:
  - "Convolutional Neural Networks" course by Andrew Ng on Coursera.
  - "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition" book by Aurélien Géron.
  - "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Ronneberger et al.






