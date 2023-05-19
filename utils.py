import tensorflow as tf
from keras_flops import get_flops


def layer_CONV_FLOPs_calculator(layer:tf.keras.layers.Conv2D)->int:
    '''
    this function calculate the FLOPs for a single CONV layer
    following the equation in
    https://indico.cern.ch/event/917049/contributions/3856417/attachments/2034165/3405345/Quantized_CNN_LLP.pdf
    :param layer:a tensorflow conv layer
    :return:
    int value represent the amount of FLOPs
    '''
    output_shape=layer.output_shape[1:]
    H_out,W_out=output_shape[0:-1]
    C_out =layer.filters
    K_h,K_w=layer.kernel_size
    C_in=layer.input_shape[-1]
    N=H_out*W_out
    M=K_h*K_w*C_in*C_out
    return (2*(N*M)+(C_out*H_out*W_out))

def layer_CONV_MAC_calculator(layer:tf.keras.layers.Conv2D)->int:
    '''
    this function calculate the MACs for a single CONV layer
    following the equation in
    https://indico.cern.ch/event/917049/contributions/3856417/attachments/2034165/3405345/Quantized_CNN_LLP.pdf
    :param layer:a tensorflow conv layer
    :return:
    int value represent the amount of MACs
    '''
    C_out =layer.filters
    K_h,K_w=layer.kernel_size
    C_in=layer.input_shape[-1]
    M=K_h*K_w*C_in*C_out
    return (M)


def layer_DENSE_MAC_calculator(layer:tf.keras.layers.Dense)->int:
    '''
    this function calculate the MACs for a single Dense layer
    following the equation in
    https://indico.cern.ch/event/917049/contributions/3856417/attachments/2034165/3405345/Quantized_CNN_LLP.pdf
    :param layer:a tensorflow Dense layer
    :return:
    int value represent the amount of MACs
    '''
    input_shape=layer.input_shape[-1]
    output_shape=layer.output_shape[-1]
    return input_shape*output_shape

def layer_DENSE_FLOPs_calculator(layer:tf.keras.layers.Dense):
    '''
    this function calculate the FLOPs for a single Dense layer
    following the equation in
    https://indico.cern.ch/event/917049/contributions/3856417/attachments/2034165/3405345/Quantized_CNN_LLP.pdf
    :param layer:a tensorflow Dense layer
    :return:
    int value represent the amount of FLOPs
    '''

    output_shape=layer.output_shape[-1]
    MAC=layer_DENSE_MAC_calculator(layer)

    return (2*MAC+(output_shape))

def get_respective_fields(model:tf.keras.Sequential):
    '''
    this function calculate the receptive  field for single path network,
    taking the model cnn layers only as for Dense layer it's receptive  field
    is the Whole image  due to the Flatten layer before it and the naturel of neuron
    the main equation can be found in
    https://rubikscode.net/wp-content/uploads/2020/05/receptive-field-formula-2.jpg

    :param model:tensorflow sequential model
    :return:
    output : list contain list of [string,int] represent the layer name and the receptive field
    '''
    number_of_layers=len(model.layers)
    r=[ 0 for _ in range(number_of_layers+1)]
    output=[]
    r[0]=1
    for l in  range(0,number_of_layers):
        s=1
        if "conv" not in model.layers[l].name: continue
        for i in range(0,l+1):
            s= model.layers[i].strides[0] * s
        r[l+1]= r[l]+(model.layers[l].kernel_size[0] - 1 )*s
        output.append([model.layers[l].name ,r[l+1]])
    print(output)
    return output


def model_FLOPSs_algorithm(model:tf.keras.Sequential)->int:
    '''
    over all function that calculate the FLOPs for sequential model
    :param model: tensorflow.keras.Sequential model
    :return:
    total_flops= total number of flops , dtype int
    '''
    total_flops=0
    for layer in model.layers:
        if "conv" in layer.name :
            print(layer.name,layer_CONV_FLOPs_calculator(layer))
            total_flops+=layer_CONV_FLOPs_calculator(layer)
        if "dense" in layer.name:
            print(layer.name,layer_DENSE_FLOPs_calculator(layer))
            total_flops+=layer_DENSE_FLOPs_calculator(layer)


    print("FLOPS CONV DENSE :",total_flops)
    return total_flops

def model_MACs_algorithm(model:tf.keras.Sequential)->int:
    '''
    over all function that calculate the MACs for sequential model
    :param model: tensorflow.keras.Sequential model
    :return:
    total_flops= total number of flops , dtype int
    '''
    total_macs=0
    for layer in model.layers:
        if "conv" in layer.name :
            print(layer.name,layer_CONV_MAC_calculator(layer))
            total_macs+=layer_CONV_MAC_calculator(layer)
        if "dense" in layer.name:
            print(layer.name,layer_DENSE_MAC_calculator(layer))
            total_macs+=layer_DENSE_MAC_calculator(layer)


    print("MAC CONV DENSE :",total_macs)
    return total_macs



def model_FLOPs_Keras(model:tf.keras.Sequential):
    '''
    keras function used to calculate  FLOPs
    this function is used to make sure that the algorithm functions works right
    :param model: tensorflow.keras.Sequential model
    :return:
    '''
    flops = get_flops(model, batch_size=1)

    print(f"FLOPS: {flops} ")# / 10 ** 9:.05

if __name__ == '__main__':
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28,28,1),name="input_layer"),
        tf.keras.layers.Flatten(name='Flatten'),
        tf.keras.layers.Dense(units=100,activation='relu',name='dense1'),
    ])
    model_FLOPSs_algorithm(model)
    model_FLOPSs_algorithm(model)
