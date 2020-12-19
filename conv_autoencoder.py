# Import all the required Libraries

import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


class conv_Autoencoder:

    '''
    class for implementing convolutional autoencoder

    '''

    def __init__(self, input_shape = None, summary = True):
        '''
        Constructor function for defining the model
        '''

        self.input_shape = input_shape
        self.summary = summary  # for printing model summary

        # ENCODER
        input = keras.Input(shape = self.input_shape, name = 'original_image')  # input
        e = Conv2D(16, (3, 3), activation='relu', padding='same')(input)
        e = MaxPooling2D((2, 2), padding='same')(e)
        e = BatchNormalization()(e)
        e = Conv2D(8, (3, 3), activation='relu', padding='same')(e)
        e = MaxPooling2D((2, 2), padding='same')(e)
        e = BatchNormalization()(e)
        e = Conv2D(8, (3, 3), activation='relu', padding='same')(e)
        e = MaxPooling2D((2, 2), padding='same')(e)
        e = BatchNormalization()(e)

        latent = Flatten(name = 'latent_features')(e)   # LATENT REPRESENTATION
        
        #DECODER
        d = Reshape((4,4,8))(latent)
        d = BatchNormalization()(d)    
        d = UpSampling2D((2, 2))(d)
        d = Conv2D(8, (3, 3), activation='relu', padding='same')(d)
        d = BatchNormalization()(d)
        d = UpSampling2D((2, 2))(d)
        d = Conv2D(16, (3, 3), activation='relu')(d)
        d = BatchNormalization()(d)
        d = UpSampling2D((2, 2))(d)
        output = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name = 'reconstructed-image')(d)   # output
        
        self._autoencoder_model = keras.Model(input, output, name = 'Autoencoder_model')  # autoencoder model
        self._encoder_model = keras.Model(input, latent, name = 'Encoder_model')  # only encoder model
        
        if self.summary:
            self._autoencoder_model.summary()


    def train(self, train_data = None, val_data = None, test_dat = None, epochs = 12, batch_size = 32):

        '''
        Train function
        '''
        
        callbacks_ = [
        keras.callbacks.ModelCheckpoint(filepath = 'weights.hdf5', monitor='val_loss', verbose=2, save_best_only=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.5,
                                            patience=3,
                                            cooldown=1,
                                            min_lr=0.00001,
                                            verbose=1)
        ]
    #  keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=7),
        
    #   History() ]
        
        self._autoencoder_model.compile(optimizer = 'adam', loss = 'binary_crossentropy')


        for e in range(epochs):


            self._autoencoder_model.fit(train_data, train_data, 
                                                initial_epoch = e,
                                                epochs = e+1,
                                                batch_size = batch_size,
                                                validation_data = (val_data, val_data),
                                                shuffle = True,
                                                callbacks = callbacks_)
            
            self.plotting(test = test_dat, saved_epoch_number = e+1)

        self._autoencoder_model.load_weights('weights.hdf5')


    def Evaluation(self, test_data = None):

        '''
        Evaluation of results
        '''
        
        self._autoencoder_model.evaluate(test_data, test_data)

    
    def predict(self, test_data = None):

        '''
        Predicting the latent_representations as well as encoded image
        '''

        latent_predictions = self._encoder_model.predict(test_data)
        reconstructed_predictions = self._autoencoder_model.predict(test_data)

        return latent_predictions, reconstructed_predictions

    def plotting(self, test = None, saved_epoch_number = None):

        '''
        PLotting fn for visualising generated images at some fixed period of time (here every 10 epochs)

        '''

        if(saved_epoch_number%10==1): # after every 5 time stamps
            _, decoded_imgs = self.predict(test_data = test)

            n = 10
            plt.figure(figsize=(20, 4))
            plt.title('Original vs Reconstructed')
            for i in range(1, n + 1):
                # Display original
                ax = plt.subplot(2, n, i)
                plt.imshow(x_test[i].reshape(28, 28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # Display reconstruction
                ax = plt.subplot(2, n, i + n)
                plt.imshow(decoded_imgs[i].reshape(28, 28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)  
            plt.savefig('images'+'epochs'+str(saved_epoch_number)+'.png')
            plt.show()


if __name__  == "__main__":


    Autoencoder = conv_Autoencoder(input_shape = (28,28,1), summary = True)

    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

    x_train, x_val = train_test_split(x_train, test_size=0.25, random_state=42)

    x_val.shape

    Autoencoder.train(train_data = x_train, val_data= x_val, test_dat = x_test, epochs = 50, batch_size = 64)

    Autoencoder.Evaluation(test_data = x_test)