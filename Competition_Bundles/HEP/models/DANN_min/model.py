import os
from sys import path
import numpy as np
import pandas as pd
from math import sqrt, log
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import pickle



from keras.models import Sequential
# ------------------------------
# Absolute path to submission dir
# ------------------------------
submissions_dir = os.path.dirname(os.path.abspath(__file__))
path.append(submissions_dir)

from systematics import postprocess


from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.models import save_model
from tensorflow.keras import backend as K
# ------------------------------
# Constants
# ------------------------------
EPSILON = np.finfo(float).eps

hist_analysis_dir = os.path.dirname(submissions_dir)
path.append(hist_analysis_dir)

from hist_analysis import compute_result

# ------------------------------
# Gradient Reversal model
# ------------------------------

import os

import tensorflow as tf


@tf.custom_gradient
def grad_reverse(x, scale=0.2):
#def grad_reverse(x, scale=1.):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy * scale
    return y, custom_grad

class GradReverse(tf.keras.layers.Layer):
    def __init__(self):
        super(GradReverse, self).__init__()

    def call(self, x):
        return grad_reverse(x)

# ------------------------------
# Baseline Model
# ------------------------------
class Model():
    """
    This is a model class to be submitted by the participants in their submission.

    This class should consists of the following functions
    1) init : initialize a classifier
    2) fit : can be used to train a classifier
    3) predict: predict mu_hats,  delta_mu_hat and q1,q2

    Note:   Add more methods if needed e.g. save model, load pre-trained model etc.
            It is the participant's responsibility to make sure that the submission 
            class is named "Model" and that its constructor arguments remains the same.
            The ingestion program initializes the Model class and calls fit and predict methods
    """

    def __init__(
            self,
            train_set=None,
            systematics=None
    ):
        """
        Model class constructor

        Params:
            train_set:
                labelled train set
                
            systematics:
                systematics class

        Returns:
            None
        """

        # Set class variables from parameters
        self.train_set = train_set
        self.systematics = systematics

        # Intialize class variables
        self.validation_sets = None
        self.theta_candidates = np.arange(0.4, 0.95, 0.02)
        self.threshold = 0.7
        self.bins = 1
        self.scaler = StandardScaler()
        self.mu_scan = np.linspace(0, 4, 100)
        self.plot_count = 2
        self.variable = "DER_deltar_lep_had"


    def fit(self):
        """
        Params:
            None

        Functionality:
            this function can be used to train a model using the train set

        Returns:
            None
        """


        self._generate_validation_sets()
        self._init_model()
        self._train()
        self._predict_holdout()
        # self._choose_theta()
        self.mu_hat_calc()
        # self._validate()
        # self._compute_validation_result()
        self._save_model()

    def predict(self, test_set):
        """
        Params:
            None

        Functionality:
           to predict using the test sets

        Returns:
            dict with keys
                - mu_hat
                - delta_mu_hat
                - p16
                - p84
        """

        print("[*] - Testing")
        test_df = test_set['data']
        test_df = self.scaler.transform(test_df)
        Y_hat_test = self._return_score(test_df)

        print("[*] - Computing Test result")
        weights_test = test_set["weights"].copy()

        # print(f"[*] --- total weight test: {weights_test.sum()}") 
        # print(f"[*] --- total weight train: {weights_train.sum()}")
        # print(f"[*] --- total weight mu_cals_set: {self.holdout['weights'].sum()}")

        weight_clean = weights_test[Y_hat_test > self.threshold]
        test_df = test_set['data'][Y_hat_test > self.threshold]
        test_array = Y_hat_test[Y_hat_test > self.threshold]

        # test_array = test_df[self.variable]

        # get n_roi\

        test_hist ,_ = np.histogram(test_array,
                    bins=self.bins, density=False, weights=weight_clean)


        mu_hat, mu_p16, mu_p84 = compute_result(test_hist,self.alpha_fun_dict,SYST=True)
        delta_mu_hat = mu_p84 - mu_p16

        mu_hat = mu_hat - self.calibration
        mu_p16 = mu_p16 - self.calibration
        mu_p84 = mu_p84 - self.calibration


        print(f"[*] --- mu_hat: {mu_hat}")
        print(f"[*] --- delta_mu_hat: {delta_mu_hat}")
        print(f"[*] --- p16: {mu_p16}")
        print(f"[*] --- p84: {mu_p84}")

        return {
            "mu_hat": mu_hat,
            "delta_mu_hat": delta_mu_hat,
            "p16": mu_p16,
            "p84": mu_p84
        }

    def _init_model(self):
        print("[*] - Intialize Baseline Model (NN bases Uncertainty Estimator Model)")


        self.input_dim = self.train_set["data"].shape[1]

        n_hidden_inv = 3; n_hidden_inv_R = 3
        n_nodes_inv = 4; n_nodes_inv_R = 4
        hp_lambda = 50

        inputs = Input(shape=(self.input_dim,))

        Dx = Dense(n_nodes_inv, activation="relu")(inputs)
        for _ in range(n_hidden_inv -1):
            Dx = Dense(n_nodes_inv, activation='relu', kernel_regularizer='l2')(Dx)

        middle_point = Dx

        for _ in range(n_hidden_inv -1):
            Dx = Dense(n_nodes_inv, activation='relu', kernel_regularizer='l2')(Dx)

        Dx = Dense(1, activation="sigmoid", name="Clf")(Dx)

        # inv_model = KerasModel(inputs=inputs, outputs=Dx)

        GRx = GradReverse()(middle_point)
        Rx = Dense(n_nodes_inv_R, activation="relu")(GRx)
        for i in range(n_hidden_inv_R -1):
            Rx = Dense(n_nodes_inv_R, activation='relu', kernel_regularizer='l2')(Rx)

        #Rx = Dense(1, activation="sigmoid")(Rx)
        Rx = Dense(1, activation="linear", name="Adv")(Rx)

        self.model = KerasModel(inputs=inputs, outputs=[Dx, Rx])

        print("[*] ---- Compiling Model")

        self.model.compile(loss=["binary_crossentropy", "mean_squared_error"], loss_weights=[1,hp_lambda], optimizer="RMSProp")

    def _generate_holdout_sets(self):
        print("[*] - Generating Validation sets")

        # Calculate the sum of weights for signal and background in the original dataset
        signal_weights = self.train_set["weights"][self.train_set["labels"] == 1].sum()
        background_weights = self.train_set["weights"][self.train_set["labels"] == 0].sum()

        # Split the data into training and holdout sets while preserving the proportion of samples with respect to the target variable
        train_df, holdout_df, train_labels, holdout_labels, train_weights, holdout_weights =  train_test_split(
            self.train_set["data"],
            self.train_set["labels"],
            self.train_set["weights"],
            test_size=0.5,
            stratify=self.train_set["labels"]
        )



        # Calculate the sum of weights for signal and background in the training and holdout sets
        train_signal_weights = train_weights[train_labels == 1].sum()
        train_background_weights = train_weights[train_labels == 0].sum()

        holdout_signal_weights = holdout_weights[holdout_labels == 1].sum()
        holdout_background_weights = holdout_weights[holdout_labels == 0].sum()

        # Balance the sum of weights for signal and background in the training and holdout sets
        train_weights[train_labels == 1] *= signal_weights / train_signal_weights
        train_weights[train_labels == 0] *= background_weights / train_background_weights

        holdout_weights[holdout_labels == 1] *= signal_weights / holdout_signal_weights
        holdout_weights[holdout_labels == 0] *= background_weights / holdout_background_weights

        train_df = train_df.copy()
        train_df["weights"] = train_weights
        train_df["labels"] = train_labels
        train_df = postprocess(train_df)

        train_weights = train_df.pop('weights')
        train_labels = train_df.pop('labels')
        

        holdout_df = holdout_df.copy()
        holdout_df["weights"] = holdout_weights
        holdout_df["labels"] = holdout_labels

        holdout_df = postprocess(holdout_df)

        holdout_weights = holdout_df.pop('weights')
        holdout_labels = holdout_df.pop('labels')

        self.train_df = train_df

        self.train_set = {
            "data": train_df,
            "labels": train_labels,
            "weights": train_weights,
            "settings": self.train_set["settings"]
        }

        self.holdout = {
                "data": holdout_df,
                "labels": holdout_labels,
                "weights": holdout_weights
            }

        
        train_signal_weights = train_weights[train_labels == 1].sum()
        train_background_weights = train_weights[train_labels == 0].sum()

        holdout_set_signal_weights = holdout_weights[holdout_labels == 1].sum()
        holdout_set_background_weights = holdout_weights[holdout_labels == 0].sum()

        print(f"[*] --- original signal: {signal_weights} --- original background: {background_weights}")
        print(f"[*] --- train signal: {train_signal_weights} --- train background: {train_background_weights}")
        print(f"[*] --- holdout_set signal: {holdout_set_signal_weights} --- holdout_set background: {holdout_set_background_weights}")

    def _train(self):

        tes_sets = []
        tes_set = self.train_set['data'].copy()

        tes_set = pd.DataFrame(tes_set)

        tes_set["weights"] = self.train_set["weights"]
        tes_set["labels"] = self.train_set["labels"]
        tes_set["tes"] = 1.0

        # tes_set = tes_set.sample(frac=0.5, replace=True, random_state=0).reset_index(drop=True)

        tes_sets.append(tes_set)

        for i in range(0, 2):

            tes_set = self.train_set['data'].copy()

            tes_set = pd.DataFrame(tes_set)

            tes_set["weights"] = self.train_set["weights"]
            tes_set["labels"] = self.train_set["labels"]

            # tes_set = tes_set.sample(frac=0.2, replace=True, random_state=i+1).reset_index(drop=True)

            # adding systematics to the tes set
            # Extract the TES information from the JSON file
            # tes = round(np.random.uniform(0.9, 1.10), 2)
            if i==0:
                tes = 0.5
            else:
                tes = 1.5

            syst_set = tes_set.copy()
            data_syst = self.systematics(
                data=syst_set,
                verbose=0,
                tes=tes
            ).data

            data_syst = data_syst.round(3)
            tes_set = data_syst.copy()
            tes_set['tes'] = 1.0
            tes_sets.append(tes_set)
            del data_syst
            del tes_set

        tes_sets_df = pd.concat(tes_sets)

        train_tes_data = (tes_sets_df).copy()

        tes_label = train_tes_data.pop('labels').array
        tes_label = np.array(tes_label).T
        # tes_label_1_temp = tes_label_1.array

        print("[*] --- tes_label_1: ", tes_label)
        tes_syst = train_tes_data.pop('tes').array
        tes_syst = np.array(tes_syst).T
        # tes_label_2_temp = tes_label_2.array
        print("[*] --- tes_label_2: ", tes_syst)
    
        # tes_label = [tes_label, tes_syst]

        # tes_label = np.array(tes_label).T
        tes_weights = train_tes_data.pop('weights').array
        tes_weights = np.array(tes_weights).T

        weights_train = tes_weights.copy()

        class_weights_train = (weights_train[tes_label == 0].sum(), weights_train[tes_label == 1].sum())

        for i in range(len(class_weights_train)):  # loop on B then S target
            # training dataset: equalize number of background and signal
            weights_train[tes_label == i] *= max(class_weights_train) / class_weights_train[i]
            # test dataset : increase test weight to compensate for sampling

        print("[*] --- Training Model")
        train_tes_data = self.scaler.fit_transform(train_tes_data)

        print("[*] --- shape of train tes data", train_tes_data.shape)

        self._fit(train_tes_data, tes_label, tes_syst, weights_train)

        # print("[*] --- Predicting Train set")
        # self.train_set['predictions'] = (self.train_set['data'], self.threshold)

        # self.train_set['score'] = self._return_score(self.train_set['data'])

        # auc_train = roc_auc_score(
        #     y_true=self.train_set['labels'],
        #     y_score=self.train_set['score'],
        #     sample_weight=self.train_set['weights']
        # )
        # print(f"[*] --- AUC train : {auc_train}")
        
        del self.train_set['data']


    def _fit(self, X, Y, Z, w):
        print("[*] --- Fitting Model") 
        self.model.fit(x=X, y=[Y,Z], sample_weight=w, epochs=4, batch_size=2*1024, verbose=1)


    def _return_score(self, X):
        y_predict = self.model.predict(X)
        y_predict = y_predict.pop(0)
        y_predict = y_predict.ravel()
        print("[*] --- y_predict: ", y_predict)
        return np.array(y_predict)
    
    def _predict_holdout(self):
        print("[*] --- Predicting Holdout set")
        X_holdout = self.holdout['data']
        X_holdout_sc = self.scaler.transform(X_holdout)
        self.holdout['score'] = self._return_score(X_holdout_sc)
        print("[*] --- Predicting Holdout set done")
        print("[*] --- score = ", self.holdout['score'])

    def mu_hat_calc(self):  
        holdout_array = self.holdout['score']
        weights_holdout = self.holdout['weights']

        # compute gamma_roi

        self.control_bins = int(self.bin_nums * (1 - self.threshold))

        holdout_hist , bins = np.histogram(holdout_array,
                    bins = self.bins, density=False, weights=weights_holdout)
        
        holdout_hist_control = holdout_hist[-self.control_bins:]

        self.theta_function()

        mu_hat, mu_p16, mu_p84, alpha = compute_result(holdout_hist_control,self.fit_function_dict_control,SYST=True)

        self.calibration = mu_hat - 1
        
        print(f"[*] --- mu_hat: {mu_hat} --- mu_p16: {mu_p16} --- mu_p84: {mu_p84} --- alpha: {alpha}")



    def nominal_histograms(self,theta):

        X_holdout = self.holdout['data'].copy()
        X_holdout['weights'] = self.holdout['weights'].copy()
        X_holdout['labels'] = self.holdout['labels'].copy()

        holdout_syst = self.systematics(
            data=X_holdout.copy(),
            tes=theta
        ).data

        label_holdout = holdout_syst.pop('labels')
        weights_holdout = holdout_syst.pop('weights')

        X_holdout_sc = self.scaler.transform(holdout_syst)
        holdout_val = self._return_score(X_holdout_sc)

        weights_holdout_signal = weights_holdout[label_holdout == 1]
        weights_holdout_background = weights_holdout[label_holdout == 0]

        holdout_signal_hist , self.bins = np.histogram(holdout_val[label_holdout == 1],
                    bins= self.bins, density=False, weights=weights_holdout_signal)
        
        holdout_background_hist , self.bins = np.histogram(holdout_val[label_holdout == 0],
                    bins= self.bins, density=False, weights=weights_holdout_background)


        return holdout_signal_hist , holdout_background_hist


    def theta_function(self,plot_count=0):

        fit_line_s_list = []
        fit_line_b_list = []
        self.coef_b_list = []
        self.coef_s_list = []
        theta_list = np.linspace(0.9,1.1,3)  
        s_list = [[] for _ in range(self.bins)]
        b_list = [[] for _ in range(self.bins)]
        
        for theta in tqdm(theta_list):
            s , b = self.theta_fuction(theta)   # ??

            for i in range(len(s)):
                s_list[i].append(s[i])
                b_list[i].append(b[i])

        print(f"[*] --- s_list shape: {np.array(s_list).shape}")
        print(f"[*] --- b_list shape: {np.array(b_list).shape}")
        print(f"[*] --- theta_list shape: {np.array(theta_list).shape}")

        for i in range(len(s_list)):
            s_array = np.array(s_list[i])
            b_array = np.array(b_list[i])

            coef_s = np.polyfit(theta_list,s_array,1)

            coef_b = np.polyfit(theta_list,b_array,1)

            self.coef_s_list.append(coef_s)
            self.coef_b_list.append(coef_b)

            fit_line_s_list.append(np.poly1d(coef_s))
            fit_line_b_list.append(np.poly1d(coef_b))


        if plot_count > 0:
            for i in range(min(plot_count,len(s_list))):
                plt.plot(theta_list,s_list[i])
                plt.show()

                plt.plot(theta_list,b_list[i])
                plt.show()


        print(f"[*] --- fit_line_s_list: {fit_line_s_list}")
        print(f"[*] --- fit_line_b_list: {fit_line_b_list}")

        self.alpha_fun_dict = {
            "gamma_roi": fit_line_s_list,
            "beta_roi": fit_line_b_list
        }


    def _save_model(self):
        print("[*] - Saving Model")
        save_model(self.model, os.path.join(submissions_dir, 'model.keras'))
        pickle.dump(self.scaler, open(os.path.join(submissions_dir, 'scaler.pkl'), 'wb'))
        settings = {
            "threshold": self.threshold,
            "calibration": self.calibration,
            "control_bins": self.control_bins,
            "bin_nums": self.bins,
            "coef_s_list": self.coef_s_list,
            "coef_b_list": self.coef_b_list
        }
        pickle.dump(settings, open(os.path.join(submissions_dir, 'settings.pkl'), 'wb'))




