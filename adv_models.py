import numpy as np

from sys import platform
from helpers import evaluate_results, read_data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras import backend as K
import keras
from keras import layers
import matplotlib.pyplot as plt

SEPARATOR = '\\' if platform == 'win32' else '/'


class ModelClass:
    def __init__(self, data: dict, NN_layers=2, NN_size=32, epochs=2, batch_size=10, seed=1, batch_norm=False) -> None:
        self.datasets = data
        self.layers = NN_layers
        self.layer_size = NN_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.batch_norm = batch_norm

    def run_models(self):
        results = {}
        print(self.datasets.keys())

        for dataset_name in self.datasets.keys():
            print(f"############# DATASET NAME AND METHOD: {dataset_name} ############")
            df = self.datasets[dataset_name]["data"].copy()  # copy dataframe

            categorical = df.select_dtypes('category')

            df[categorical.columns] = categorical.apply(LabelEncoder().fit_transform)
            target = self.datasets[dataset_name]["target"]

            if self.datasets[dataset_name]["pred_type"] == "regression":
                ### TODO: adapt model
                print("regression")
            elif self.datasets[dataset_name]["pred_type"] == "classification":
                ### TODO: adapt model
                print("classification")
            else:  # both, run all
                raise TypeError("Prediction type not supported")

            X = df.drop([target], axis=1)
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)

            y_pred, history = self.train_NN_model(X_train, X_test, y_train, y_test,
                                                  self.datasets[dataset_name]["pred_type"])


            results[dataset_name] = {"pred_type": self.datasets[dataset_name]["pred_type"]}
            results[dataset_name]['NN'] = {'pred': list(y_pred), 'true': list(y_test),
                                           'train_loss': history.history['loss'],
                                           'val_loss': history.history['val_loss']}

            if self.datasets[dataset_name]["pred_type"] == "classification":
                results[dataset_name]['NN']['train_acc'] = history.history['accuracy']
                results[dataset_name]['NN']['val_acc'] = history.history['val_accuracy']

            # self.train_AE_model(X_train, X_test, y_train, y_test, self.datasets[dataset_name]["pred_type"])
            # self.train_VAE_model(X_train, X_test, y_train, y_test, self.datasets[dataset_name]["pred_type"])

        return results

    def evaluate(self, y_test, y_pred, type="regression"):
        if type == "regression":
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print("The model performance for testing set")
            print("--------------------------------------")
            print('MAE is {}'.format(mae))
            print('MSE is {}'.format(mse))
            print('R2 score is {}'.format(r2))
        else:
            print("Classification report")
            print(classification_report(y_test, y_pred))
            print("Confusion matrix")
            print(confusion_matrix(y_test, y_pred))
            print("Accuracy score")
            print(accuracy_score(y_test, y_pred))

    def train_NN_model(self, X_train, X_test, y_train, y_test, pred_type="classification"):
        """
        Here we train the models and get access to data for evaluation
        
        """
        print("**********NN************")
        try:
            from keras.models import Sequential
            from keras.layers import Dense
            import tensorflow as tf
            from tensorflow.keras.layers import BatchNormalization
        except ImportError:
            print('Keras and TF could not be imported')
            return None

        model = Sequential()

        size = self.layer_size
        model.add(Dense(size, input_dim=X_train.shape[1], activation='relu'))

        for i in range(1, self.layers - 1):
            size = int(size / 2)
            model.add(Dense(size, activation='relu'))
            if self.batch_norm:
                model.add(BatchNormalization())  # adding some batch norm if we have specified it
        # model.add(Dense(1, activation='sigmoid'))  #!for regression we dont want a sigmoid so add relu instead below

        if pred_type == "classification":
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        else:
            model.add(Dense(1, activation='relu'))
            model.compile(loss=tf.keras.losses.MeanAbsolutePercentageError(),
                          optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                          metrics=[tf.keras.metrics.MeanSquaredError()])
        his = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=self.epochs,
                        batch_size=self.batch_size)

        y_pred = model.predict(X_test)

        if (pred_type == "classification"):
            y_pred = np.where(np.array(y_pred) > 0.5, 1, 0)

        evaluate_results(y_test, y_pred, model_type=pred_type)
        print("************************")

        return y_pred, his

    def train_VAE_model(self, X_train, X_test, y_train, y_test, pred_type="classification"):
        print("********** VAE ************")
        try:
            from keras.models import Sequential
            from keras.layers import Dense
            import tensorflow as tf
        except ImportError:
            print('Keras and TF could not be imported')
            return None

        original_dim = X_train.shape[1]
        intermediate_dim = 8
        latent_dim = 2

        inputs = keras.Input(shape=(original_dim,))
        h = layers.Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = layers.Dense(latent_dim)(h)
        z_log_sigma = layers.Dense(latent_dim)(h)

        # z = layers.Lambda(self.sampling)([z_mean, z_log_sigma])

        z = self.sampling(z_mean, z_log_sigma, latent_dim)

        # Create encoder
        encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

        # Create decoder
        latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
        x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = layers.Dense(original_dim, activation='sigmoid')(x)
        decoder = keras.Model(latent_inputs, outputs, name='decoder')

        # Create model with custom loss
        outputs = decoder(encoder(inputs)[2])
        vae = keras.Model(inputs, outputs, name='vae')

        reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')

        history = vae.fit(X_train, X_train,
                          epochs=self.epochs,
                          batch_size=self.batch_size,
                          shuffle=True,
                          validation_data=(X_test, X_test))

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        # Evaluation is done by loss and plotting the results on 2D map if we have a classification problem
        if pred_type == "classification":
            # TODO: Probably the output of these results should go somewhere else
            x_test_encoded = encoder.predict(X_test, batch_size=self.batch_size)
            x_test_encoded = np.array(x_test_encoded)[0]
            plt.figure(figsize=(6, 6))
            plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
            plt.colorbar()
            plt.show()

        # _, accuracy = evaluate_results(X_train, y_train)
        # print('Accuracy: %.2f' % (accuracy * 100))

        print("************************")

        return []

    def sampling(self, z_mean, z_log_sigma, latent_dim):
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=0.1)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def train_AE_model(self, X_train, X_test, y_train, y_test, pred_type="classification"):
        """
        Here we train the models and get access to data for evaluation
        
        """
        print("********** AE ************")
        try:
            from keras.models import Sequential
            from keras.layers import Dense
            import tensorflow as tf
        except ImportError:
            print('Keras and TF could not be imported')
            return None

        # Create encoder
        input = keras.Input(shape=X_train.shape[1], )
        encoded = layers.Dense(32, activation='relu')(input)
        encoded = layers.Dense(8, activation='relu')(encoded)
        encoded = layers.Dense(2, activation='relu')(encoded)
        encoder = keras.Model(input, encoded, name='encoder')

        # latent_input = keras.Input(shape=(2,), name='latent_input')

        # Create decoder
        decoded = layers.Dense(8, activation='relu')(encoded)
        decoded = layers.Dense(32, activation='relu')(decoded)
        decoded = layers.Dense(X_train.shape[1], activation='sigmoid')(decoded)
        # decoder = keras.Model(latent_input, decoded, name='decoder')

        # output = decoder(encoder(input)[2])

        autoencoder = keras.Model(input, decoded, name='autoencoder')
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        # Train
        history = autoencoder.fit(X_train, X_train,
                                  epochs=self.epochs,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  validation_data=(X_test, X_test))

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        # Evaluation is done by loss and plotting the results on 2D map if we have a classification problem
        if pred_type == "classification":
            # TODO: Probably the output of these results should go somewhere else
            x_test_encoded = encoder.predict(X_test, batch_size=self.batch_size)
            x_test_encoded = np.array(x_test_encoded)
            plt.figure(figsize=(6, 6))
            plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
            plt.colorbar()
            plt.show()

        print("************************")

        return []


def run_advanced_models(args, dataset):
    data = read_data(dataset)
    if data is None:
        return "No data available for dataset"
    else:
        print("Running advance models for dataset {}".format(dataset))
        models = ModelClass(data, seed=args.seed, batch_norm=args.batchnorm)
        results = models.run_models()
        return results
