# Multi_layer_perceptron_mnist
Creating Multi-layer-perceptron using Mnist data


Overview: 

1 . Takes inputs as a matrix (2D array of numbers)
2.Multiplies the input by a set weights (performs a dot product aka matrix multiplication)
3.Applies an activation function
4.Returns an output
5.Error is calculated by taking the difference from the desired output from the data and the predicted output. This creates our gradient descent, which we can use to alter the weights
6.The weights are then altered slightly according to the error.
7.To train, this process is repeated 1,000+ times. The more the data is trained upon, the more accurate our outputs will be.


Implementation :

Implemented the forward propagation and compute the loss.
X- input dataset , y is the true label vector, l-r learning rate of the optimization, epochs is the number of times the we training the training data , parameters learned by the model , then they can be used to predict.The update rule uses learning rate of 0.1. The code is run for 2000 epochs. The activation function is sigmoid  and Softmax of the form f(x) = 1 / 1 + exp(-x) and e / e.sum() Its range is between 0 and 1 . and this values are passed to derivative of softmax function to error.The performance of classification is measured using confusion matrix. The code computes specificity, sensitivity, precision, recall, accuracy and F-score. The dataset contains of digits where we are training the data predict the data to improve the prediction and accuracy of the data. Predict method is used to predict the data of test and train data. Back_propagation function initializes the network by assigning random weights to the connections of input, hidden and output layers. The train_network trains the network by first propagating the inputs through a system of weighted connections. For each input sample, the intermediate layer computes weighted sum, derives activation function value and forwards the output of the neuron to the next layer. The output layer compares the expected and the obtained neuron value and back propagates the difference so that the weight adjustments can minimize error for the next iteration. Activate function computes the weighted sum of inputs, transfer function calculates neuron output due to the activation function, backward_propagate_error derives error at every layer using the delta rule, and update_weight stores the modified weights of neurons at each layer of the network 

Confusion_matrix uses the actual and predicted output to generate values for false positives, false negatives, true positives and true negatives. 

[[20 0 0 0 0 0 0 0 0 0] [ 0 17 0 0 0 0 0 0 0 2] [ 0 0 21 0 0 0 0 0 0 0] [ 0 0 0 20 0 0 0 1 1 0] [ 0 0 0 0 17 0 1 0 0 0] [ 0 0 0 0 0 17 0 0 0 0] [ 0 0 0 0 0 0 16 0 0 0] [ 0 0 0 0 0 0 0 21 0 0] [ 0 0 0 0 0 1 0 0 14 0] [ 0 1 0 0 0 1 0 0 0 9]] value of confusion matrix 

False Positives [0 1 0 0 0 2 1 1 1 2] False Negetives [0 2 0 2 1 0 0 0 1 2] True Positives [20 17 21 20 17 17 16 21 14 9] True Negetives [160 160 159 158 162 161 163 158 164 167] 

Sensitivity [1. 0.89473684 1. 0.90909091 0.94444444 1. 1. 1. 0.93333333 0.81818182] Specificity [1. 0.99378882 1. 1. 1. 0.98773006 0.99390244 0.99371069 0.99393939 0.98816568] 

Precision [1. 0.94444444 1. 1. 1. 0.89473684 0.94117647 0.95454545 0.93333333 0.81818182] Recall [1. 0.89473684 1. 0.90909091 0.94444444 1. 1. 1. 0.93333333 0.81818182] 

Áccuracy  [1. 0.98333333 1. 0.98888889 0.99444444 0.98888889 0.99444444 0.99444444 0.98888889 0.97777778] 

F1_Score [1. 0.91891892 1. 0.95238095 0.97142857 0.94444444 0.96969697 0.97674419 0.93333333 0.81818182] 

Cost of train data graph :




Cost of test data graph:


calculated accuracy with actual and predicted values.
The train set accuracy is 95.4% and test set accuracy is 95.5% 
