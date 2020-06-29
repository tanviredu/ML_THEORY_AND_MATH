# KNN (*k*-nearest neighbours algorithm)

--------------



KNN is the simplest algorithm possible yet very powerful. how k classify and how kNN algorithm is trained ?its very simple , it classify by measuring distance among nearest  data point. suppose there are two different data



![img](https://raw.githubusercontent.com/tanviredu/ML_THEORY_AND_MATH/master/knn.PNG)

**"a"** and **"o"** and in the 2D plane and there is an unknown data **"C"** . how classify is the data belongs to a or **o** ?

ANS: 

calculate the distance from c to the other elements now .Then there comes a topic the value of K

suppose in the KNN the **K=3** then the algorithm will consider the nearest three elements and and you can 

See

see![img](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/KnnClassification.svg/220px-KnnClassification.svg.png)



there are two "o" and 1 a so the vote goes for the "o" so the Knn algorithm classify the unknown value as o

it can be multidimensional

**The naive version of the algorithm is easy to implement by computing the distances from the test example to all stored examples,**



we use the normal Euclidian distance for calculating  distance



Four steps to a KNN algorithm do 

1) find the distance from the target point to all other point

2) sort it based on  their distance

3) choose the first K value (k is a numeric number)

4) and then compare which has more vote

 

# 



![K Nearest Neighbor Classification - Animated Explanation for ...](https://machinelearningknowledge.ai/wp-content/uploads/2018/08/KNN-Classification.gif)





# Decision Tree Algorithm

----

Decision Tree algorithm belongs to the family of supervised learning algorithms

. How Decision Tree algorithm can Classify Data.

Suppose you have some data . first you separate based on their colour (first characters)

then based on size then based on the height now after continuously doing you are actually making a lot of branch . and at one point these value will be separated based on different criteria . now after that you dot an unknown value .then you compare it with the tree like which branch it belongs based on colour size and other properties . at the end you will find a branch that i suits. that's how decision tree classifies the data 







![img](https://miro.medium.com/max/688/1*bcLAJfWN2GpVQNTVOCrrvw.png)





**Steps in Decision Tree algorithm:**

1. It begins with the original set S as the root node.
2. On each iteration of the algorithm, it iterates through the very unused attribute of the set S and calculates **Entropy(H)** and **Information gain(IG)** of this attribute.
3. It then selects the attribute which has the smallest Entropy or Largest Information gain.
4. The set S is then split by the selected attribute to produce a subset of the data.
5. The algorithm continues to recur on each subset, considering only attributes never selected before.









# Random Forest 

Random forest is actually a  a combination of a lot of different Decision Tree. it is used because Decision tree Result can be biased. to more generalize the model This is used . like the name A lot of tree makes the forest



![img](https://miro.medium.com/max/884/1*5vlUF8FRR6flPPWK4wt-Kw.png)





Let’s look at a case when we are trying to solve a classification problem. As evident from the image above, our training data has four features- Feature1, Feature 2, Feature 3 and Feature 4. Now, each of our bootstrapped sample will be trained on a particular subset of features. For example, Decision Tree 1 will be trained on features 1 and 4 . DT2 will be trained on features 2 and 4, and finally DT3 will be trained on features 3 and 4. We will therefore have 3 different models, each trained on a different subset of features. We will finally feed in our new test data into each of these models, and get a unique prediction. The prediction that gets the maximum number of votes will be the ultimate decision of the random forest algorithm.





# Support Vector machine

----



Support Vector machine or SVM Has a different Approach for classifying . 

### It Creates a hyper plane like it you separate the data with a sheet in a 3D space

This is the 2D representation of the SVC (Support Vector Classification)





![img](https://miro.medium.com/max/600/1*Sg6wjASoZHPphF10tcPZGg.png)

 



# What happened of the data is not linearly Separable like this ?





![img](https://miro.medium.com/max/600/1*C3j5m3E3KviEApHKleILZQ.png)





# The Kernel Trick

It is not necessary that you always have to use a linear hyper plaice using polynomial equation you can make the hyper place  folded based on your need and that's the main power comes in the Support vector machine



Like This

![img](https://miro.medium.com/max/600/1*gt_dkcA5p0ZTHjIpq1qnLQ.png)











# Neural Network



Take a look at the picture



![img](https://miro.medium.com/max/443/1*FLyQXTVPePOtRpeO75-_wg.png)



A neural network is exactly what it says in the name. It is a network of neurons that are used to process information. To create these, scientists looked at the most advanced data processing machine at the time the brain. Our brains process information using networks of neurons. They receive an input, process it, and accordingly output electric signals to the neurons it is connected to. Using bio-mimicry, we were able to apply the architecture of our brains to further the field of artificial intelligence.





## there are three different main Components of neural net

1) Input Layer (take the input)

2) hidden layer (Do the processing)

3) Output Layer ( show the output)





# How the neural Net is Trained



1)  neural Net start with the random weight and bias

2)  then they train themselves over and over again (but how ?)

 [They do this by calculating the error in each prediction its called **the cost** of neural network]

3) the error is calculated by the this equation (for simple neural net)

```
(target- output)^2 
```





![img](https://miro.medium.com/max/1280/1*UjQ6E8ZCbQBKSurLqnZRkg.png)

## The Training part

The entire goal of training the neural network is to minimize the cost. Neural networks do this using a process called **backpropagation**. This seems like a complicated word but its quite simple.  forward propagation is when you run information through a neural network to give you a result. Backward propagation is literally the same thing but backward. You just start at the output layer and run the neural network in reverse to optimize the weights and biases. the  optimize means change the weight and bias so the equations can fir the data

**After it finds the pattern of the data then new data is feed to classify with the model it takes the input and say which it belongs based on the its characters**





![img](https://miro.medium.com/max/791/1*nX6L9rclTnx0Hph5i01wzg.jpeg)







# Convolutional Neural Network (CNN)





Convolutional neural network (ConvNets or CNNs) is one of the main categories to do images recognition, images classifications.



 **How ConvNet Works**

1) look at the picture



![img](https://miro.medium.com/max/1255/1*XbuW8WuRrAY5pC4t-9DZAQ.jpeg)



2) it does exactly like our eyes does i.t takes the small portion of the data and then extract the the main feature of the data and it does by taking  small part from the image and then extract the main feature of the data and by doing over and over it extract the basic characteristics of the image . you can directly just compare a image with other . the CNN take the image extract the feature by convolution and then compare the basic character like simple stroke and line and then it can classify the image.



# in the picture

the car image is selected by this

![img](https://miro.medium.com/max/268/1*MrGSULUtkXc0Ou07QouV8A.gif)



Then again and again 

## 2) after that is will be feed to a Traditional neural network to classify the image (it is showed in the classification) part of the picture





![img](https://miro.medium.com/max/948/1*4GLv7_4BbKXnpc6BRb0Aew.png)



# the process is follows

- Provide input image into convolution layer
- Choose parameters, apply filters with strides, padding if requires. Perform convolution on the image and apply ReLU activation to the matrix.
- Perform pooling to reduce dimensionality size
- Add as many convolutional layers until satisfied
- Flatten the output and feed into a fully connected layer (FC Layer)
- Output the class using an activation function (Logistic Regression with cost functions) and classifies images.