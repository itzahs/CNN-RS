# Transfer deep learning for remote sensing datasets: <br/> A comparison study

This repository contains the jupyter notebooks used in the study. <br/>
Two versions are available: one using a batch of 128 and the other one a batch of 64. <br/>
All the experiments for RGB images have been conducted in Google  Colab  Pro  with  16GB  GPU  and  25GB  RAM.

Please take care of the following three things while using the script:<br/>
1. The implemented DL architecture is ResNet50. <br/>
2. The name of the notebook references the dataset used to train the network <br/>
3. All the models are used to classify the Eurosat Dataset. <br/>

## Reference document: [link](link)

## References

Codes adapted from:
1. CodeX For ML, 18 Sept. 2020, "How to use TensorFlow Datasets? Image classification with EuroSAT dataset with TFDS", [Video], YouTube, URL: https://www.youtube.com/watch?v=6th3rahsw9Y.
2. Vera, Adonaí. 1 Dec. 2021, “Curso Profesional De Redes Neuronales Con Tensorflow.” [E-Learning Website], URL: https://platzi.com/cursos/redes-neuronales-tensorflow/. 
3. Jens Leitloff and Felix M. Riese, "Examples for CNN training and classification on Sentinel-2 data", Zenodo, 10.5281/zenodo.3268451, 2018. [Repository], URL: https://github.com/jensleitloff/CNN-Sentinel


Tensorflow models and datasets:
1. Maxim Neumann, Andre Susano Pinto, Xiaohua Zhai,and Neil Houlsby,   “In-domain representation learningfor remote sensing,” Nov. 2019. URL: https://tfhub.dev/google/collections/remote_sensing/1
