import tensorflow as tf
from tensorflow.keras import layers

conv_defaults = {"kernel_size": 3, 
                 "padding": "same", 
                 "activation": "relu", 
                 "kernel_initializer": "he_normal"}

class DownsampleBlock(layers.Layer):
    
    def __init__(self, filters, num_filters, **kwargs):
        super(DownsampleBlock, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.layers = [layers.Conv2D(filters=filters, **conv_defaults) for _ in range(num_filters)]      
        self.layers.append(layers.MaxPool2D(pool_size=2, strides=2))
        
    def call(self, inputs):
        x = inputs
        for i in range(self.num_filters):
            x = self.layers[i](x)      
        down = self.layers[-1](x)  
        return x, down

class UpsampleBlock(layers.Layer):
    
    def __init__(self, filters, num_filters, **kwargs):
        super(UpsampleBlock, self).__init__(**kwargs)
        self.layers = [layers.Conv2DTranspose(filters=filters, kernel_size=2, strides=2)]
        [self.layers.append(layers.Conv2D(filters=filters, **conv_defaults)) for _ in range(num_filters)]  
        
    def call(self, connect, inputs):
        x = self.layers[0](inputs)
        x = tf.concat([connect, x], axis=3)
        for layer in self.layers[1:]:
            x = layer(x)       
        return x
        
class UNet(tf.keras.models.Model):
    
    def __init__(self, filters, maps, **kwargs):
        super(UNet, self).__init__(**kwargs)
        self.num = len(filters)
        self.downsample = [DownsampleBlock(f, 2) for f in filters]
        self.bridge = [layers.Conv2D(filters=filters[-1]*2, **conv_defaults), 
                       layers.Conv2D(filters=filters[-1]*2, **conv_defaults)]
        self.upsample = [UpsampleBlock(f, 2) for f in filters[::-1]]
        self.output_layer = layers.Conv2D(filters=maps, kernel_size=1, activation="sigmoid")
        
    def call(self, inputs):
        
        out = inputs
        connections = []
        
        for layer in self.downsample:
            x, out = layer(out)
            connections.append(x)  
            
        for layer in self.bridge:
            out = layer(out)    
        connections = connections[::-1]
            
        for _ in range(self.num):
            out = self.upsample[_](connections[_], out)  
        out = self.output_layer(out)
        
        return out