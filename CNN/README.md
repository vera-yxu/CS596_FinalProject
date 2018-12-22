Authors: Ying Xu and Emma Westin

### Usage 

python CNNmodel.py --e 50 --o Adadelta 

 

### INPUT 

Our CNN uses matrices as its input. To get the proper representation, the DNA sequences were converted so that each base was transformed into a binary representation using the following scheme:  

        'A': [1, 0, 0, 0], 

        'T': [0, 0, 0, 1], 

        'G': [0, 0, 1, 0], 

        'C': [0, 1, 0, 0], 

        'D': [1, 0, 1, 1], 

        'N': [1, 1, 1, 1], 

        'S': [0, 1, 1, 0], 

        'R': [1, 0, 1, 0] 

 

With each sequence length containing 60 bases, the matrix results in the shape (60, 4). The class labels were converted as well; IE= 0, EI= 1, and N=2.  

### MODEL ARCHITECTURE 

We built a CNN model from scratch using Keras API with a TensorFlow backend [3,4]. Our CNN model has five layers, the first layer is a convolutional layer with one filter in size 3x4, number of rows of the filter is selected based on a frequently used k-mer (short word in length k, here k=3) approach when implementing DNA sequence comparisons, number of columns of the filter is kept as the same as sequence matrix. The output from the first convolutional layer is a 58x1 vector, and it is followed by three fully connected (dense) layers with neurons of 58, 28 and 10, the last layer is an output layer with three classes. We used ReLU activation function for the first four layers and softmax for the last output layer.  

### TRAINING AND TESTING 

We divided the data into 90% training and 10% testing, which is 2870 sequences for training and 320 sequences for testing. The model was trained using a batch size of 1, with 10 epochs as the default value. However, we implemented a system to allow for the epoch value to be changed according to user input (flag ‘--e’). The same argument parser was set to specify an optimizer to be used with flag ‘--o’. The optimizer can be selected from the following: adam, rmsprop, sgd and Adadelta. All initial learning rates are set to 0.01 except Adadelta is set to 1.0 to enable faster learning. We found the Adadelta optimizer gives the highest accuracy, so it is the default optimizer.  

### OUTPUT 

In the output, we report the test loss, and test accuracy, as well as a Confusion Matrix, and Classification report. The model also outputs two graphs of the training and validation accuracy and loss. 
