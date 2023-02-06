from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from  tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from  tensorflow.keras.optimizers import SGD, Adam
import  matplotlib.pyplot as plt

from io import StringIO
from scipy.io import arff


from sklearn.utils import  shuffle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from sklearn.model_selection import  train_test_split

#f = StringIO("sampledataset.arff")
#data = arff.loadarff("sampledataset.arff")
#print(data)


csv_path="MNISTDAE.csv"
csv_path_full="MNIST.csv"
saveName="MNISTReduced13_L3.csv"
batch_size = 128
epochs=300
test_split=0.20
mu=0.0
sig=1.3
# Add more dense filters if you want 500	335	115	50
GSD=[128,86,28]
HAR=[561,375,125,50]
spam=[499,335,115,50]
fct=[54,36,15]
IoT=[115,75,25]
kdd=[34,22,15]
mnist=[784,522,175,60]
mnist2=[784,522,60]
news=[1000,650,250,80]
news2=[1000,750,350,180,50]#[300,280,180,50]
#[1000,750,350,180,50]
dsname=news2
noofFeatures=dsname[0]
latent_dim = dsname[-1]
layer_filters = dsname[0:len(dsname)-1]

pd_dataset = pd.read_csv(csv_path,header=None)

def encoder_model(encoder_input):

    # First build the Encoder Model

    x = encoder_input

    for filters in layer_filters:
        x = Dense(filters)(x)

    # Generate the latent vector
    x = Flatten()(x)

    latent = Dense(latent_dim, name='latent_vector')(x)

    model=Model(encoder_input, latent, name='encoder')

    return model


def decoder_model(encoder_input,encoder_shape_out):

    decoder_in = encoder_input

    for in_filter in reversed(layer_filters):
        decoder_in = Dense(in_filter)(decoder_in)

    decoder_in = Dense(encoder_shape_out)(decoder_in)

    outputs = Activation('sigmoid', name='decoder_output')(decoder_in)

    model=Model(encoder_input, outputs, name='decoder')

    return model

def list_map(value):

    all_elememts=[]

    for element in value:

        current_list=list()
        current_list.append(element)

        all_elememts.append(current_list)

    return all_elememts


def generated_attributes(total):

    attribute_list=[]

    for i in range(total):

        attribute_list.append("Attribute"+str(i+1))

    attribute_list.append("class_label")

    return attribute_list

def train_dencoder_dense():

        # CSV loading dataset
        pd_dataset_features = pd_dataset.iloc[:,0:noofFeatures]
        feature_label=pd_dataset.iloc[:,noofFeatures]


        x_train, x_test = train_test_split(shuffle(pd_dataset_features),test_size=test_split)

        #normalizer = MinMaxScaler()
        #x_train = normalizer.fit_transform(x_train)
        #x_test = normalizer.transform(x_test)

        #x_train = x_train.astype('float32') / 255
        #x_test = x_test.astype('float32') / 255

        # Generate corrupted dataset by adding noise with normal dist
        # centered at 0.5 and std=0.5
        noise = np.random.normal(loc=mu, scale=sig, size=x_train.shape)
        x_train_noisy = x_train + noise


        noise = np.random.normal(loc=mu, scale=sig, size=x_test.shape)
        x_test_noisy = x_test + noise


        # Network parameters


        # Build Encoder Model

        inputs = Input(shape=(x_train_noisy.shape[1],), name='encoder_input')
        encoder = encoder_model(inputs)
        encoder.summary()

        # Build the Decoder Model
        latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
        decoder = decoder_model(latent_inputs,x_train_noisy.shape[-1])
        decoder.summary()

        # Autoencoder = Encoder + Decoder
        # Instantiate Autoencoder Model
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        autoencoder.summary()

        optimizer=SGD(lr=0.01)

        autoencoder.compile(loss='mse', optimizer=optimizer)

        cp_callback=ModelCheckpoint(
                            filepath="model_saved_dense/cp-{epoch:04d}.ckpt",
                            verbose=1,
                            save_weights_only=True,
                            period=epochs)

        csv_logger = CSVLogger('training_log.csv')


        # Train the autoencoder
        autoencoder.fit(x_train_noisy,
                        x_train,
                        validation_data=(x_test_noisy, x_test),
                        epochs=epochs,
                        batch_size=batch_size, callbacks=[cp_callback,csv_logger])

        print("Model Training Completed",end="\t")

        print("Saving latent features ...... ", end="\t")


        # Predict Autoencoder output from corrupted test data

        reconstructed_dataset = autoencoder.predict(x_test_noisy)

        save_data_to_csv(reconstructed_dataset,"test_reconstructed_data.csv")

        reduced_dimension_features=encoder.predict(x_test_noisy)

        save_data_to_csv(reduced_dimension_features, "test_reduced_features.csv")


        dataset_read_for_features = pd.read_csv(csv_path_full,header=None)

        all_data = dataset_read_for_features.iloc[:,0:noofFeatures]
        all_label= np.asarray(dataset_read_for_features.iloc[:,noofFeatures]).tolist()

        all_label_res =np.asarray(list_map(list(all_label)))


        reduced_dimension_features_all = encoder.predict(all_data)

        normalizer_all = MinMaxScaler()
        HARNorm = normalizer_all.fit_transform(all_data)#temp
        all_data_normalize = normalizer_all.fit_transform(reduced_dimension_features_all)

        reduced_features_all_with_label =np.append(all_data_normalize,all_label_res,axis=1)


        attribute_list=generated_attributes(reduced_features_all_with_label.shape[1]-1)

        data_csv = pd.DataFrame(reduced_features_all_with_label,columns=attribute_list)

        data_csv.to_csv(saveName)



def plot_graph():

    data = pd.read_csv("training_log.csv")

    epoch = np.arange(epochs) + 1

    train_loss = np.asarray(data["loss"])
    val_loss = np.asarray(data["val_loss"])

    plt.plot(epoch, train_loss, 'b', label='Training loss')
    plt.plot(epoch, val_loss, 'r', label='Validation loss')

    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def save_data_to_csv(generated_data=None,csv_name="sample.csv"):

    data=np.asarray(generated_data)

    pd.DataFrame(data).to_csv(csv_name)


if __name__ == "__main__":

    train_dencoder_dense()
    plot_graph()
