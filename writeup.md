#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report



## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network inspired by NVIDIA architecture

Model Architecture is as follows:

1) Image is cropped by removing 50 pixels and 20 pixels from both top and bottom directions respectively

2)Image normalizaion

3) Convolution layer :
		depth - 24
		filter size - 5x5
		stride - 2,2 

4) Activation Layer - PReLU
		PReLU was chosen instead of ReLU for better consideration of negative values for activation of regression problem

5) Convolution layer :
		depth - 36
		filter size - 5x5
		stride - 2,2 

6) Activation Layer - PReLU

7) Convolution layer :
		depth - 48
		filter size - 5x5
		stride - 2,2 

8) Activation Layer - PReLU

9) Convolution layer :
		depth - 64
		filter size - 3x3
		stride - 1,1 

10) Activation Layer - PReLU

11) Max Pooling Layer with 2x2 window

12) Flatten

13) Fully Connected Layer - 500 units

14) Activation Layer - PReLU

15) Fully Connected Layer - 200

16) Activation Layer - PReLU

17) Fully Connected Layer - 40

18) Activation Layer - PReLU

17) Output Layer - 1 unit or Single output

The model includes PReLU layers to introduce nonlinearityand the data is normalized in the model using a Keras lambda layer. 

####2. Attempts to reduce overfitting in the model


The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and sample data to achieve optimum mix of data.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach


My first step was to use a convolution neural network model similar to the NVIDIA architecture. I thought this model might be appropriate because it is practically tested on car which is similar to simulator. Therefore transfer learning can be achieved easily using this model

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set

The model was trained with training data

The final step was to run the simulator to see how well the car was driving around track one.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


####2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 3 laps on track one using center lane driving, 2 laps of recovery driving and 2 laps of reverse direction driving.

To augment the data sat, I also randomly flipped images and angles thinking that this would generalize the model.
To generalize the model for different lighting condition images where altered with random brightness value and a random shadow was added to a random portion of image.

The training data was collected 6 times to improve the smooth and clean turning angles. Importance of clean data was realized from this project. To ensure good data even the sample data provided was added to the training data.

Even then model failed for approx 30 tries with little modifications in the model.

At this time, it was realized that model was overfitting for straight driving since most of the training data is concentrated for center lane driving. So in code , only around 25% of close to 0 steering angle data was considered randomly for each image batch

Thus after around 60 tries, a successful model was derived 

The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by 5 epochs
