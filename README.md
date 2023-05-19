# Fashon_detection
# table of Content
1. Instalation
2. Data
3. Data Handling 
4. Model
5. Model Details
# Instalation
first thing you need to do is install the requriments 
```
pip install -r requirments.txt
```
after that the env will be ready to run 
# Data
- the data used is mnist dataset provided by tensorflow 
- the data contain 60k train images and 10k dev images 
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
- All functions used to load the data in the [data_script.py]()
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
- all results can be found in [model.py]() script
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
- the CONV layers are **Kernal = 3x3** with **Stride = 1** with **Activation = Relu **
- taking input **28x28x1** indicate that it takes **gray image**
- **Number of Parameters **
```
    # Total params: 3,909,430
    # Trainable params: 3,909,430
    # Non-trainable params: 0
```
### The Model FLOPs & MACs & Respective Field
- the **FLOPs**
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
|---|---|---|---|---|---|---|---|---|---|---|---|
