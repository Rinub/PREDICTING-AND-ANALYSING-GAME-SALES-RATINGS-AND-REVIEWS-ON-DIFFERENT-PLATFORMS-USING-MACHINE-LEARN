<h1 align="center">FACE MASK DETECTION SYSTEM USING DEEP LEARNING METHODOLOGY</h1>

<div align= "center"><p align="center"><img src="https://i.ytimg.com/vi/JRmA9Baip0o/maxresdefault.jpg" width="700" height="400"></p>
  <h4>Face Mask Detection system built with Keras/TensorFlow using Deep Learning and Computer Vision concepts in order to detect face masks in static images </h4>
</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Rinub/FACE-MASK-DETECTION-SYSTEM-USING-DEEP-LEARNING-METHODOLOGY/blob/main/README.md)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555)](https://www.linkedin.com/in/ibrahimrinub/)


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;



## Motivation
> Due to Covid-19, there are currently no effective face mask detection applications, which are in high demand for transportation, densely populated areas, residential districts, large-scale manufacturers, and other organizations.

>> The deep neural network models are used for analyzing any visual imagery. It takes the image data as input, captures all the data, and sends it to the layers of neurons. It has a fully connected layer, which processes the final output that represents the prediction about the image.

>>> After comparing all the proposed models, the model with less memory will be deployable in the embedded devices used for surveillance purposes.
 

--> If interested :email: rinubronic@gmail.com

 
## Project demo
:movie_camera: [YouTube Demo Link](https://youtu.be/wYwW7gAYyxw)

## :star: Features
 In this research, Deep Learning methodologies is been used to detect faces with and without the mask is achieved. I've constructed a 3+ model that detects face masks trained on 7553 photos with three color channels using custom CNN, VGG16, VGG19, and EfficientNetB0 (RGB). Several assessment indicators are used to compare the performance of all the models to select the best-performing model.

This system can therefore be used in real-time applications which require face-mask detection for safety purposes due to the outbreak of Covid-19. This project can be integrated with embedded systems for application in airports, railway stations, offices, schools, and public places to ensure that public safety guidelines are followed.

This image dataset is obtained from Kaggle. This image set of data included 7553 RGB images is comprised of two classes in different folders. The two folders have images of people wearing the mask and without wearing the mask respectively. The folders are classified and labeled as "with masks" and " without masks" to reduce the complication while training and testing the deep learning model. The amount of data of people wearing face masks is 3725, and the person without wearing the face mask is 3828. From Prajna Bhandary's Github account, 1776 pictures were acquired, including both images of the person wearing and without wearing a face mask. 

## :file_folder: Dataset
The dataset used can be downloaded here - [Click to Download](https://www.kaggle.com/omkargurav/face-mask-dataset)

This dataset consists of __7553 images__ belonging to two classes:
*	__with_mask: 3725 images__
*	__without_mask: 3828 images__

## Explanatory analysis of Dataset

<p align="center"><img src="https://www.pyimagesearch.com/wp-content/uploads/2020/04/face_mask_detection_dataset.jpg" width="700" height="400"></p>


## :warning: TechStack/framework used

- [OpenCV](https://opencv.org/)
- [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [EfficientNetB0](https://arxiv.org/abs/1801.04381)
- [VGG16](https://arxiv.org/abs/1801.04381)
- [VGG19](https://arxiv.org/abs/1801.04381)



---
### Packages Installation & Execution

> Run these commands after cloning the project

| Commands                                                                                                                     | Time to completion |
|------------------------------------------------------------------------------------------------------------------------------|--------------------|
| sudo apt install -y libatlas-base-dev liblapacke-dev gfortran                                                                | 1min               |
| sudo apt install -y libhdf5-dev libhdf5-103                                                                                  | 1min               |
| pip3 install -r requirements.txt                                                                                             | 1-3 mins           |
| wget "https://raw.githubusercontent.com/PINTO0309/Tensorflow-bin/master/tensorflow-2.4.0-cp37-none-linux_armv7l_download.sh" | less than 10 secs  |
| ./tensorflow-2.4.0-cp37-none-linux_armv7l_download.sh                                                                        | less than 10 secs  |
| pip3 install tensorflow-2.4.0-cp37-none-linux_armv7l.whl                                                                     | 1-3 mins           |
