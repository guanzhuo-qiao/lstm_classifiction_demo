# LSTM model on stocks classification 

We construct a RNN(LSTM) model to detect the stocks' next 3 month performance in the future using the 100 daily history of the factor. This is only a simple demo and the accuracy of the model is not very high. There are till many issues need to be addressed. Here we present the basic structure of the model. 

## data preparation

We use stock "ABT" data from 2010/01/01 to 2020/01/01 daily stock prices (HLOC) and we use these data to compute 9 technical indicators (William R, percentage price oscillator, Percentage Volume Oscillator, Stochastic Oscillator, MACD, Bollinger Bands and RSI). These 9 indicators are served as factors and will be used to predict the stocks classification. 

The label of the stocks are the rank of risk adjusted return (Sharpe Ratio) in the next 62 days (one season). If we separate the Sharpe Ratio value of the stocks into 5 layers, we transfer the problem to a classification problem.  In other words, we try to use the last 100 day's factors value to predict the next 1 season stocks' performance.

After computing the labels and factors, we delete the nan data and normalize the factors data.

![Capture1](C:\Users\Qiao Guanzhuo\Desktop\Capture1.PNG)

![Capture2](C:\Users\Qiao Guanzhuo\Desktop\Capture2.PNG)

Then we will use these data to fit a LSTM model.

## Construct the model

We construct a LSTM model using the TensorFlow Keras API. The model contains 2 LSTM layers with dropout rate of 0.2 and 1 dense layer with a "softmax" activation function. We use RMSprop optimizer and train the model for 30 epochs and each epoch train for 200 steps. We separate the data set into training set and test set. There are 2412 data points and train set has 1400 points and the rest are belongs to test set.

![Capture3](C:\Users\Qiao Guanzhuo\Desktop\Capture3.PNG)

## Fit the data and get the results

We fit our data into the model and get the accuracy of the train set and test set.

The model and reach 87.59% of accuracy but only has a 17.11% on test set. 

![image-20200121182645390](C:\Users\Qiao Guanzhuo\AppData\Roaming\Typora\typora-user-images\image-20200121182645390.png)

The above is the accuracy of two data sets through the training steps.

![image-20200121182936196](C:\Users\Qiao Guanzhuo\AppData\Roaming\Typora\typora-user-images\image-20200121182936196.png)

Here is the confusion matrix of the model prediction in the test set. We can see that the model tend to classify the stock into the third layer.

![image-20200121183212300](C:\Users\Qiao Guanzhuo\AppData\Roaming\Typora\typora-user-images\image-20200121183212300.png)

If we plot the time series of the prediction and real data we hardly find the relationship between this two.

We can calculate the correlation of real label and predicted label and we find the small.

```python
In[33]: np.corrcoef(y_val_single,y_pre)
Out[33]: 
array([[ 1.       , -0.0816057],
       [-0.0816057,  1.       ]])
```

If we focus on the change of the label, we then compute the correlation of the one lag difference of the data.

```python
In[34]: np.corrcoef(np.diff(y_val_single),np.diff(y_pre))
Out[34]: 
array([[ 1.        , -0.02224983],
       [-0.02224983,  1.        ]])

```

They are still little correlated with each other.

## Problems

1) When we use some fiscal data, they are almost updated in season and can hardly avoid looking-ahead bias in back testing. How to perform that data.

2) How to compute the impact of other stocks. We only conduct an experiment only on one stock, and if we consider other stock's influence, how to achieve that in code?

3) When we label the data using the ranking information, we loss some absolute value information. What's more, it will be a jump in label when we doing this. In other words, if two stocks has very similar performance, but they are laying at two sides of the criteria then we will label them into two different class, however the distance between them are not that far.

4) Do we got enough data? This should also consider the training time. If we have a deep network and long looking back, it will take a lot of time to train the model.

5) Is there any other way we can use to do the same prediction?







 