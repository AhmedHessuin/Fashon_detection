import tensorflow as tf
from utils import model_FLOPSs_algorithm,model_MACs_algorithm,get_respective_fields

def model_arc(number_of_classes:int=10)->tf.keras.Sequential:
    '''
    this function build a simple Sequential model with the shape of mnist as input shape
    due to the simplicity of the training data the method used is Sequential model, the
    model is compiled with adam optimizer, with SCC (SparseCategoricalCrossentropy) loss

    :param
    number_of_classes: the number of class we will predict mainly for this task is 10
    :return:
    tensorflow.keras.Sequential model
    '''
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28,28,1),name="input_layer"),
        tf.keras.layers.Conv2D(filters=20,kernel_size=3,strides=1,padding='valid',activation='relu',name="conv1"),
        tf.keras.layers.Conv2D(filters=40,kernel_size=3,strides=1,padding='valid',activation='relu',name="conv2"),
        tf.keras.layers.Conv2D(filters=80,kernel_size=3,strides=1,padding='valid',activation='relu',name="conv3"),
        tf.keras.layers.Flatten(name='Flatten'),
        tf.keras.layers.Dense(units=100,activation='relu',name='dense1'),
        tf.keras.layers.Dense(number_of_classes,activation='softmax',name='dense_output_layer')
    ])
    adam=tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=adam,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


def model_arc2(number_of_classes:int=10)->tf.keras.Sequential:
    '''
    this function build a simple Sequential model with the shape of mnist as input shape
    due to the simplicity of the training data the method used is Sequential model, the
    model is compiled with adam optimizer, with SCC (SparseCategoricalCrossentropy) loss

    :param
    number_of_classes: the number of class we will predict mainly for this task is 10
    :return:
    tensorflow.keras.Sequential model
    '''
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28,28,1),name="input_layer"),
        tf.keras.layers.Conv2D(filters=20,kernel_size=3,strides=1,padding='valid',activation='relu',name="conv1"),
        tf.keras.layers.Conv2D(filters=40,kernel_size=3,strides=2,padding='valid',activation='relu',name="conv2"),
        tf.keras.layers.Conv2D(filters=80,kernel_size=3,strides=2,padding='valid',activation='relu',name="conv3"),
        tf.keras.layers.Flatten(name='Flatten'),
        tf.keras.layers.Dense(units=100,activation='relu',name='dense1'),
        tf.keras.layers.Dense(number_of_classes,activation='softmax',name='dense_output_layer')
    ])
    adam=tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=adam,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model





if __name__ == '__main__':
    print("normal model")
    model=model_arc(10)
    model.summary()
    # Total params: 3,909,430
    # Trainable params: 3,909,430
    # Non-trainable params: 0
    model_FLOPSs_algorithm(model) # output number of FLOPs 44 237 550
    # conv1    256880
    # conv2   8317440
    # conv3  27917120 # most layer CONV 3 because the max number of input features in this layer
    # dense1  7744100
    # dense_output_layer 2010
    model_MACs_algorithm(model)
    # conv1      180
    # conv2     7200
    # conv3    28800
    # dense1 3872000
    # dense_output_layer 1000
    # MAC CONV DENSE : 3909180

    get_respective_fields(model)
    # [['conv1', 3], ['conv2', 5], ['conv3', 7]]
    # can be increases with max pooling or dilated conv, or stack more conv layers
    print('-='*20)

    print("FLOPs optimized model")
    model=model_arc2(10)
    model.summary()
    # Total params: 237,430
    # Trainable params: 237,430
    # Non-trainable params: 0
    model_FLOPSs_algorithm(model)#output number of FLOPs 4 180 350
    # conv1   256880
    # conv2  2079360
    # conv3  1442000 # most layer CONV 3 because the max number of input features in this layer
    # dense1  400100
    # dense_output_layer 2010
    model_MACs_algorithm(model)
    # conv1     180
    # conv2    7200
    # conv3   28800
    # dense1 200000
    # dense_output_layer 1000
    # MAC CONV DENSE : 237180
    get_respective_fields(model)
    # [['conv1', 3], ['conv2', 7], ['conv3', 15]]
    # in this model we used stride 2 to increase the respective field and reduce the number of FLOPs


