import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_training_data()->[np.ndarray,np.ndarray]:
    '''
    this function load the training data of fashion mnist dataset
    :return:
    list contain (train images, train labels), dtype: numpy ndarry
    '''
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels),(_,_) = fashion_mnist.load_data()
    train_images=np.expand_dims(train_images,-1)
    return [train_images, train_labels]

def load_dev_data()->[np.ndarray,np.ndarray]:
    '''
    this function load the dev data of fashion mnist dataset
    :return:
    list contain (dev images, dev labels), dtype: numpy ndarry
    '''
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (_,_),(test_images, test_labels)=fashion_mnist.load_data()
    test_images=np.expand_dims(test_images,-1)

    return [test_images, test_labels]


if __name__ == '__main__':

    fashion_mnist = tf.keras.datasets.fashion_mnist
    train_images, train_labels=load_training_data()
    test_images, test_labels=load_dev_data()

    print("train images shape ",train_images.shape)
    print("train labels shape ",train_labels.shape)
    print("test images shape ",test_images.shape)
    print("test labels shape ",test_labels.shape)

    print("Number of Training images for each class")
    label_dict={
        0:0,
        1:0,
        2:0,
        3:0,
        4:0,
        5:0,
        6:0,
        7:0,
        8:0,
        9:0
    }
    for label in train_labels:
        label_dict[label]+=1

    print(label_dict)
    print("-"*20)
    number_of_samples_to_vis=10

    print("VIS TRAINING DATA")
    for i in range(number_of_samples_to_vis):
        img=train_images[i]
        label=train_labels[i]
        cv2.imshow("img",img)
        print("label",label)
        cv2.waitKey()

    print("VIS DEV DATA")
    for i in range(number_of_samples_to_vis):
        img=test_images[i]
        label=test_labels[i]
        cv2.imshow("img",img)
        print("label",label)
        cv2.waitKey()

    total_number_of_classes=10
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    class_labels = [ 0, 1, 2, 3, 4,
                     5, 6, 7, 8, 9]


    #vis with plot
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

