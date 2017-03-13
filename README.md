Read more about it on my [website](https://bhaktipriya96.wordpress.com/dense-optical-flow-prediction/) 
As a part of the Statistical Methods of Artificial Intelligence Course Project, under the mentorship of Professor Avinash Sharma, we built a deep learning mechanism to predict the dense optical flow of every pixel given a static image. This approach considers non-semantic form of action prediction that makes it domain independent and makes no assumptions of the underlying scene. Most of research in this area has looked at different aspects of the problem.

*   Prediction focused on predicting the trajectory for the given input image.
*   Prediction focused on more semantic forms of prediction: what is going to happen next.
*   We aim at a much richer form of prediction in terms of pixels or the features of the next frame.

We used a CNN model with the VGG architecture to predict the dense optical flow.  The classical approach to such problems is that of regression. However, here the optical flow prediction problem is modeled as a classification problem. We performed k-means clustering of optical flow vectors and quantized them in 40 major directions. We trained the network by giving it the images, and per pixel optimal flow labels. Thus, given a scene our network can predict which objects will move and in what direction they will move. We trained our model on the HMDB-51 dataset, which consists of human day to day actions. Professor Abhinav Gupta's work in this domain has been very instrumental during this project. [gallery ids="1091,1089" type="rectangular"]

## Preprocessing

![screenshot-from-2017-01-21-001730](https://bhaktipriya96.files.wordpress.com/2017/01/screenshot-from-2017-01-21-001730.png)

## Regression to Classification

The classical approach to such problems is that of regression. However, here the optical flow prediction problem is modeled as a classification problem. We performed k-means clustering of optical flow vectors and quantized them in 40 major directions. ![screenshot-from-2017-01-21-001753](https://bhaktipriya96.files.wordpress.com/2017/01/screenshot-from-2017-01-21-001753.png)

## Training

We train our Convolution neural network by providing it with the image and per pixel optical flow labels. We used the Gunnar Farneback algorithm for calculating the optical flow. ![screenshot-from-2017-01-21-001805](https://bhaktipriya96.files.wordpress.com/2017/01/screenshot-from-2017-01-21-001805.png)

## The Architecture

We used the standard VGG architecture. We used a pretrained model and fine tuned it to solve our classification problem. [gallery ids="1084,1073" type="rectangular"]

## Results

We trained and tested our network on HMDB-51 dataset and used a cross validation scheme of 50:50 to test our method. We obtained an accuracy of around 83%.
