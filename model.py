from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import concatenate, Conv2DTranspose, BatchNormalization, Dropout, Activation, Add
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, BatchNormalization, Activation, Multiply, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Reshape
from keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from keras.initializers import he_normal
from keras import backend as K



def conv_block(x, num_filters, kernel_size, padding="same", act=True):
    x = Conv2D(num_filters, kernel_size, padding=padding, use_bias=False)(x)
    x = BatchNormalization()(x)
    if act:
        x = Activation("relu")(x)
    return x

def spatial_attention(x):
    avg_pool = GlobalAveragePooling2D()(x)
    max_pool = GlobalMaxPooling2D()(x)
    pool = Concatenate()([avg_pool, max_pool])
    attention = Dense(units=1, activation='sigmoid')(pool)
    attention = Reshape((1, 1, 1))(attention)
    scaled = Multiply()([x, attention])
    return scaled

def multires_block(x, num_filters, alpha=1.67):
    W = num_filters * alpha
    x0 = x
    x1 = conv_block(x0, int(W*0.167), 3)
    x2 = conv_block(x1, int(W*0.333), 3)
    x3 = conv_block(x2, int(W*0.5), 3)
    xc = Concatenate()([x1, x2, x3])
    xc = spatial_attention(xc)
    xc = BatchNormalization()(xc)
    nf = int(W*0.167) + int(W*0.333) + int(W*0.5)
    sc = conv_block(x0, nf, 1, act=False)
    x = Activation("relu")(xc + sc)
    x = BatchNormalization()(x)
    return x

def res_path(x, num_filters, length):
    for i in range(length):
        x0 = x
        x1 = conv_block(x0, num_filters, 3, act=False)
        sc = conv_block(x0, num_filters, 1, act=False)
        attention = tf.keras.layers.GlobalAveragePooling2D()(x1)
        attention = tf.keras.layers.Dense(num_filters, activation='sigmoid')(attention)
        attention = tf.keras.layers.Reshape((1, 1, num_filters))(attention)
        x1 = tf.keras.layers.Multiply()([x1, attention])
        x = tf.keras.layers.Add()([x1, sc])
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
    return x

def encoder_block(x, num_filters, length):
    x = multires_block(x, num_filters)
    s = res_path(x, num_filters, length)
    p = MaxPooling2D((2, 2))(x)
    return s, p

def decoder_block(x, skip, num_filters):
    x = Conv2DTranspose(num_filters, 2, strides=2, padding="same")(x)
    x = Concatenate()([x, skip])
    x = multires_block(x, num_filters)
    return x

def build_multiresunet(input_shape):
    inputs = Input(input_shape)
    s1, p1 = encoder_block(inputs, 32, 4)
    s2, p2 = encoder_block(p1, 64, 3)
    s3, p3 = encoder_block(p2, 128, 2)
    s4, p4 = encoder_block(p3, 256, 1)
    b1 = multires_block(p4, 512)
    d1 = decoder_block(b1, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    model = Model(inputs, outputs, name="MultiResUNET")
    return model