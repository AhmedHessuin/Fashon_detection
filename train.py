import tensorflow as tf
import os,datetime
import tqdm
import numpy as np
from data_script import load_training_data,load_dev_data
import cv2
from model import model_arc,model_arc2

def call_backs(save_every=1,exp_name="fashon"):
    es = tf.keras.callbacks.EarlyStopping(monitor='accuracy', verbose=1, patience=5)
    os.makedirs(f"checkpoints/{exp_name}",exist_ok=True)
    mc = tf.keras.callbacks.ModelCheckpoint(os.path.join(f"checkpoints/{exp_name}", f'{exp_name}_'+'{epoch:04d}.hdf5'),
                                            verbose=1, save_weights_only=False, period= save_every)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.1,
                                                     patience=3, min_lr=0.000001,verbose=1)

    logdir = os.path.join(f"log/{exp_name}", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logdir,exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    return [mc,es, reduce_lr, tensorboard_callback]


if __name__ == '__main__':
    train_images, train_labels=load_training_data()
    test_images, test_labels =load_dev_data()
    model=model_arc2(10)
    exp_name='fast_model'
    history=model.fit(train_images, train_labels, epochs=10, batch_size=32,callbacks=call_backs(exp_name=exp_name))
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    f1=open(f'checkpoints/{exp_name}/Test_Accuracy.txt','w')
    f1.write(f'test_acc : {test_acc}\n')


    hit=0
    miss=0
    total=0

    gt_dict={
        0:{"TP":0,'FP':0,"FN":0,"Name":'T-shirt/top'},
        1:{"TP":0,'FP':0,"FN":0,"Name":'Trouser'},
        2:{"TP":0,'FP':0,"FN":0,"Name":'Pullover'},
        3:{"TP":0,'FP':0,"FN":0,"Name":'Dress'},
        4:{"TP":0,'FP':0,"FN":0,"Name":'Coat'},
        5:{"TP":0,'FP':0,"FN":0,"Name":'Sandal'},
        6:{"TP":0,'FP':0,"FN":0,"Name":'Shirt'},
        7:{"TP":0,'FP':0,"FN":0,"Name":'Sneaker'},
        8:{"TP":0,'FP':0,"FN":0,"Name":'Bag'},
        9:{"TP":0,'FP':0,"FN":0,"Name":'Ankle boot'},
    }
    for idx,img in enumerate(tqdm.tqdm(test_images)):
        img = (np.expand_dims(img,0))
        result=model.predict(img)
        highest_score=np.argmax(result[0])
        if highest_score != test_labels[idx]:
            miss+=1
            gt_dict[test_labels[idx]]['FN']+=1
            gt_dict[highest_score]["FP"]+=1
        else:
            hit+=1
            gt_dict[test_labels[idx]]['TP']+=1

        total+=1
    f1.write(f"total images = {total}\n"
             f"right = {hit}\n"
             f"wrong = {miss}\n"
             f"ACC   = {hit/total}\n"
             f"-------\n")
    f1.write("PRECISION X RECALL\n")
    for key in gt_dict.keys():
        f1.write(f"{gt_dict[key]['Name']}\n"
                 f"TP : {gt_dict[key]['TP']}\n"
                 f"FP : {gt_dict[key]['FP']}\n"
                 f"FN : {gt_dict[key]['FN']}\n"
                 f"PRECISION : {gt_dict[key]['TP']/(gt_dict[key]['TP']+gt_dict[key]['FP'])}\n"
                 f"RECALL    : {gt_dict[key]['TP']/(gt_dict[key]['TP']+gt_dict[key]['FN'])}\n"
                 f"---------\n"
                 )
    f1.close()

