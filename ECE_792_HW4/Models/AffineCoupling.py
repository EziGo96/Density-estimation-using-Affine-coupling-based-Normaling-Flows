'''
Created on 06-Apr-2023

@author: EZIGO
'''
from tensorflow import keras
from tensorflow.keras import regularizers

class AffineCoupling:
    @staticmethod
    def build(input_shape,neurons,reg):
        input = keras.layers.Input(shape=input_shape)
        '''translation'''
        t_layer_1 = keras.layers.Dense(neurons, kernel_regularizer=regularizers.l2(reg))(input)
        t_layer_1 =keras.layers.PReLU()(t_layer_1)
        t_layer_2 = keras.layers.Dense(input_shape, activation="linear", kernel_regularizer=regularizers.l2(reg))(t_layer_1)
        
        '''scaling'''
        s_layer_1 = keras.layers.Dense(neurons, kernel_regularizer=regularizers.l2(reg))(input)
        s_layer_1 =keras.layers.PReLU()(s_layer_1)
        s_layer_2 = keras.layers.Dense(input_shape, activation="tanh", kernel_regularizer=regularizers.l2(reg))(s_layer_1)
    
        return keras.Model(inputs=input, outputs=[s_layer_2, t_layer_2],name="AffineCoupling")
            
            
            
            
        
        
        