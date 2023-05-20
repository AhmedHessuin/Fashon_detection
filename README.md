# Fashon_detection
# table of Content
1. Instalation
2. Data
3. Data Handling 
4. Model
5. Model Details
6. the Model Accuracy and Metric
7. Discussion
# Instalation
first thing you need to do is install the requriments 
```
conda create -n Fashon_Classification python=3.8 
conda activate Fashon_Classification
conda install -c anaconda cudatoolkit=11.2
pip install -r requirments.txt
```
after that the env will be ready to run 
# Data
- the data used is fashion_mnist dataset provided by tensorflow 
- the data contain 60k train images and 10k dev images, for train 6k image for each class and 1k in the dev
- the images are Gray lvl with (28x28)
- the data contain labels as sprase for 10 different types of cloths 
```
'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
```
with annotation like 
```
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```
this is for the data section 
# Data Handling 
1. download the training data with 
```
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels),(_,_) = fashion_mnist.load_data()
```
2. change the image channels to be (28,28,1), to use CNN layers
```
train_images=np.expand_dims(train_images,-1)
```
3. doing the same to the test data 
```
fashion_mnist = tf.keras.datasets.fashion_mnist
(_,_),(test_images, test_labels)=fashion_mnist.load_data()
test_images=np.expand_dims(test_images,-1)
```
- All functions used to load the data in the [data_script.py](data_script.py)
# Model
because of the simplicity of the training the model archtecture is so simple, and the method of building the archtecture was **Sequential**
Starting with Simple model ARC
```
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28,28,1),name="input_layer"),
        tf.keras.layers.Conv2D(filters=20,kernel_size=3,strides=1,padding='valid',activation='relu',name="conv1"),
        tf.keras.layers.Conv2D(filters=40,kernel_size=3,strides=1,padding='valid',activation='relu',name="conv2"),
        tf.keras.layers.Conv2D(filters=80,kernel_size=3,strides=1,padding='valid',activation='relu',name="conv3"),
        tf.keras.layers.Flatten(name='Flatten'),
        tf.keras.layers.Dense(units=100,activation='relu',name='dense1'),
        tf.keras.layers.Dense(number_of_classes,activation='softmax',name='dense_output_layer')
    ])
```
this model is so simple contain only 3 conv layers followed by 1 dense then another dense layer for the output 
# Models Details
- in this section we will talk about the models used and the different between each model
- the FLOPs for each model 
- the MACs for each model
- all results can be found in [model.py](model.py) script
## First Model
### The Model ARCH
```
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28,28,1),name="input_layer"),
        tf.keras.layers.Conv2D(filters=20,kernel_size=3,strides=1,padding='valid',activation='relu',name="conv1"),
        tf.keras.layers.Conv2D(filters=40,kernel_size=3,strides=1,padding='valid',activation='relu',name="conv2"),
        tf.keras.layers.Conv2D(filters=80,kernel_size=3,strides=1,padding='valid',activation='relu',name="conv3"),
        tf.keras.layers.Flatten(name='Flatten'),
        tf.keras.layers.Dense(units=100,activation='relu',name='dense1'),
        tf.keras.layers.Dense(number_of_classes,activation='softmax',name='dense_output_layer')
    ])
```
- the CONV layers are **Kernal = 3x3** with **Stride = 1** with **Activation = Relu**
- taking input **28x28x1** indicate that it takes **gray image**
- **Number of Parameters**
```
    # Total params: 3,909,430
    # Trainable params: 3,909,430
    # Non-trainable params: 0
```
### The Model FLOPs & MACs & Respective Field
- the **FLOPs**
- the equation from ![](https://snipboard.io/TcSLNO.jpg)
- [link](https://indico.cern.ch/event/917049/contributions/3856417/attachments/2034165/3405345/Quantized_CNN_LLP.pdf)
- aslo confirmed that i am doing it right with `from keras_flops import get_flops`
```
    # conv1    256880
    # conv2   8317440
    # conv3  27917120 # most layer CONV 3 because the max number of input features in this layer
    # dense1  7744100
    # dense_output_layer 2010
    # total number of FLOPs 44 237 550
```
- the **MACs**
```
    # conv1      180
    # conv2     7200
    # conv3    28800
    # dense1 3872000 # MOST MACs
    # dense_output_layer 1000
    # total number of MACs : 3 909 180
```
- the **Respective Field**
- ![](https://rubikscode.net/wp-content/uploads/2020/05/receptive-field-formula-2.jpg)
```
[['conv1', 3], ['conv2', 5], ['conv3', 7]]
```
## Second Model
### The Model ARCH
```
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28,28,1),name="input_layer"),
        tf.keras.layers.Conv2D(filters=20,kernel_size=3,strides=1,padding='valid',activation='relu',name="conv1"),
        tf.keras.layers.Conv2D(filters=40,kernel_size=3,strides=2,padding='valid',activation='relu',name="conv2"),
        tf.keras.layers.Conv2D(filters=80,kernel_size=3,strides=2,padding='valid',activation='relu',name="conv3"),
        tf.keras.layers.Flatten(name='Flatten'),
        tf.keras.layers.Dense(units=100,activation='relu',name='dense1'),
        tf.keras.layers.Dense(number_of_classes,activation='softmax',name='dense_output_layer')
    ])
```
- the CONV layers are **Kernal = 3x3** with **Stride = 2** with **Activation = Relu** except first CONV use **Stride = 1**
- taking input **28x28x1** indicate that it takes **gray image**
- **Number of Parameters **
```
    # Total params: 237,430
    # Trainable params: 237,430
    # Non-trainable params: 0
```
- note that the number of parameters is reduced because the size of the input layer for the **First Dense** is reduced thanks to the **Stride = 2 **
### The Model FLOPs & MACs & Respective Field
- the **FLOPs**
- the equation from ![](https://snipboard.io/TcSLNO.jpg) 
- aslo confirmed that i am doing it right with `from keras_flops import get_flops`
- [link](https://indico.cern.ch/event/917049/contributions/3856417/attachments/2034165/3405345/Quantized_CNN_LLP.pdf)
```
    # conv1   256880
    # conv2  2079360
    # conv3  1442000 # most layer CONV 3 because the max number of input features in this layer
    # dense1  400100
    # dense_output_layer 2010
    # total number of FLOPs 4 180 350
```
- note that the number of FLOPs is reduced because the **Stride = 2** effect on the input size for each layer
- the **MACs**
```
    # conv1     180
    # conv2    7200
    # conv3   28800
    # dense1 200000 # MOST MACs
    # dense_output_layer 1000
    # total number of MACs : 237 180
```
- note the number of MACs in the most heavy layer **dense1** is reduced because the input size is reduced thanks to the **Stride = 2** effect
- the **Respective Field**
- ![](https://rubikscode.net/wp-content/uploads/2020/05/receptive-field-formula-2.jpg)
```
[['conv1', 3], ['conv2', 7], ['conv3', 15]]
```
- note the recptive field change in **conv2** and **conv3** because of the **Stride = 2** 

## the Model Accuracy and Metric
- Mainly used SparseCategoricalCrossentropy as a loss 
- Mainly used **Precision** and **Recall** for Accuracy metric and **HIT RATE (over all right answers / total number of data)**
  - choosed this metric because they indicate very well the behaviour of a **Classification Model**
  - **Precision**: how many right answers i get for each class if the model made prediction of this class
  - **Recall** : how many right answer i get for each class from all the Predicted Data  
  - **HIT RATE** : how good the model over all with all classes 


### Results of the both Models
 
|Model Name|T-shirt/top|Trouser|Pullover|Dress|Coat|Sandal|Shirt|Sneaker|Bag|Ankle boot|Acc|
|--------|---|---|---|---|---|---|---|---|---|---|---|
|Model_1(stride_1)|PRECISION:0.798 & RECALL:0.866|PRECISION:0.981 & RECALL:0.973|PRECISION:0.851 & RECALL:0.804|PRECISION:0.923 & RECALL:0.854|PRECISION:0.781 & RECALL:0.867|PRECISION:0.982 & RECALL:0.957|PRECISION:0.703 & RECALL:0.67|PRECISION:0.921 & RECALL:0.973|PRECISION:0.968 & RECALL:0.963|PRECISION:0.971 & RECALL:0.941|0.8868%|
|Model_2(stride_2)|PRECISION:0.855 & RECALL:0.836|PRECISION:0.981 & RECALL:0.976|PRECISION:0.789 & RECALL:0.867|PRECISION:0.879 & RECALL:0.907|PRECISION:0.850 & RECALL:0.75|PRECISION:0.968 & RECALL:0.97|PRECISION:0.715 & RECALL:0.699|PRECISION:0.949 & RECALL:0.952|PRECISION:0.949 & RECAL:0.978|PRECISION:0.957 & RECALL:0.963|0.8898%|
|---|---|---|---|---|---|---|---|---|---|---|---|
- You can get the result from [train.py](train.py) as it save the **Chekpoints** while training and **Test** after training done, saving **Logs** in log folder
# Discussion
in this we will discuss 
## Approch 
- downloading the data and see the number of label for each class
- Some Data Analysis
    - knowing the image size
    - knowing the color domain 
    - looking about the data distribution for each class 
    - looking for outlier  
- building the model
    - looking about the data and how complex is the target 
    - based on the target complex the model is decided to be kinda right for example, not using resnet 152 for 28x28x1 images 
    - because complex model with simple data can lead to a bad result
        - first train with small and avg model 
        - increase the model Parameters and experiment it 
        - decrease the model parameters and experiment it
        - after some experiments you will find the best model for this task  
    - try good optimizer and train with different Learning Rate ( as it can be critical )
    - compile the model and see it's FLOPs & MACs and Parameters 
    - see if there is a way to reduce the FLOPs & MACs
    - this is why i choosed model 2 instead of model 1 because model 2 achieve same result and has min FLOPs 
    - can try more smaller model or only double layer model with 2 Dense layers (this won't achieve the task requirments for (Respective Field)
- Loss
    - deciding the loss is a critical decision as based on your decision the model can change a lot 
    - see the type of loss that meet the requriment of the target 
    - using the loss for Classification (CrossEntropy) and decide the activation function of the last layer (softmax)
    
- Train
    - write a call backs with tensorboard to trace the model training and save checkpoint every 1 epoch
- Test 
    - testing the model with the Metric i choosed ( Precision and Recall ) with the default metric (Acc) 
    - saving the result in text file to be found later 

## How to reduce number of FLOPs
this can be done by 
- more conv layers with stride 2 
    - as we can see in model with Stride 2 the FLOPs got reduced
- Pooling 
    - that will reduce the number of operations because the input size is reduced  
- Separable Convolutions
    - **Spatially Separable Convolutions**doing like Inception Net breaking the 3x3 to (1x3) (3x1) 
    - **Depthwise Separable Convolutions** Depthwise convolution followed by Pointwise convolution
- Reduce model Size 

## How to increase the Respective Field
this can be done by 
- stacking more layer 
    - as we stack more layers the Respective Field increase ( this can be consuming )
- Pooling
    - Pooling or increasing the stride of the CONV increase the Respective FIeld as we saw in the model with Stride 2
- Dilated Conv
    - as it's using wilder kernal size with the same Parameters 
    


