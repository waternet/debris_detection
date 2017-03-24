# debris_detection
Python code for debris detection using logistic regression on pictures

# What is it?
This is the first code to try and create a machine learning model (neural network) that is able to recognize floating debris from normal water. The current code scores 97% accurate but this of course all depends on the amount of pictures and the quality of the labels which is so far not good enough. 

![Screenshot](https://github.com/waternet/debris_detection/blob/master/screenie.png)

# TODO
* Add a lot more training data (so far 26 label 1 and approx 120 label 0 which is not at all adequate!)
* Better train and test code
* Testing with other learning rates
* Transformation to convolutional neural network
* Build for Tensorflow
* Build for streaming data
* Find some kind of way to preprocess the images and seperate water from other stuff (different camera?)

# The whole story
The Nautonomous project was started by Waternet in Amsterdam. The Nautonomous is meant to be a autonomous boat and the original first task is to collect floating debris from the canals of Amsterdam. There is a lot of development going on with respect to the autonomous driving of the ship. Apart from that software there should also be a way to use cameras and detect the floating debris. That is where this code is meant for. The development scheme is;

* collect video material of floating debris
* create labeled training data
* show that debris detection is possible using python and neural networks
* create Tensorflow implementation
* implement debris detection module into the ROS code of the Nautonomous

# Who do I talk to?
Rob van Putten, rob.van.putten@waternet.nl, breinbaasnl@gmail.com
