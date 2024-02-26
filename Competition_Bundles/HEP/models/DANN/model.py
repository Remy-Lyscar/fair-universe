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


from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.models import save_model
from tensorflow.keras import backend as K
# ------------------------------
# Constants
# ------------------------------
EPSILON = np.finfo(float).eps

hist_analysis_dir = os.path.dirname(submissions_dir)
path.append(hist_analysis_dir)

from hist_analysis import calculate_comb_llr

# ------------------------------
# Gradient Reversal model
# ------------------------------

import os
# reduce number of threads
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
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
        self.best_theta = 0.7
        self.bins = 1
        self.scaler = StandardScaler()
        self.mu_scan = np.linspace(0, 4, 100)
        self.plot_count = 2
        self.calibration = None

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
        # self._save_model()

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
        weights_train = self.train_set["weights"].copy()
        weights_test = test_set["weights"].copy()

        print(f"[*] --- total weight test: {weights_test.sum()}") 
        print(f"[*] --- total weight train: {weights_train.sum()}")
        print(f"[*] --- total weight mu_cals_set: {self.holdout['weights'].sum()}")

        weight_clean = weights_test[Y_hat_test > self.best_theta]
        test_df = test_set['data'][Y_hat_test > self.best_theta]

        # test_array = test_df['DER_deltar_lep_had']
        
        
        # get n_roi
        n_roi = (weight_clean.sum())

        mu_hat = (n_roi - self.beta_roi)/self.gamma_roi

        sigma_mu_hat = np.sqrt(n_roi)/self.gamma_roi

        delta_mu_hat = 2*sigma_mu_hat

        mu_p16 = mu_hat - sigma_mu_hat
        mu_p84 = mu_hat + sigma_mu_hat


        print(f"[*] --- mu_hat: {mu_hat.mean()}")
        print(f"[*] --- delta_mu_hat: {delta_mu_hat}")
        print(f"[*] --- p16: {mu_p16}")
        print(f"[*] --- p84: {mu_p84}")

        return {
            "mu_hat": mu_hat.mean(),
            "delta_mu_hat": delta_mu_hat,
            "p16": mu_p16,
            "p84": mu_p84
        }

    def _init_model(self):
        print("[*] - Intialize Baseline Model (NN bases Uncertainty Estimator Model)")


        self.input_dim = self.train_set["data"].shape[1]

        n_hidden_inv = 4; n_hidden_inv_R = 4
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

    def _generate_validation_sets(self):
        print("[*] - Generating Validation sets")

        print("[*] --- train_set features: ", self.train_set["data"].columns)

        # Calculate the sum of weights for signal and background in the original dataset
        signal_weights = self.train_set["weights"][self.train_set["labels"] == 1].sum()
        background_weights = self.train_set["weights"][self.train_set["labels"] == 0].sum()

        # Split the data into training and validation sets while preserving the proportion of samples with respect to the target variable
        train_df, valid_df, train_labels, valid_labels, train_weights, valid_weights = train_test_split(
            self.train_set["data"],
            self.train_set["labels"],
            self.train_set["weights"],
            test_size=0.2,
            stratify=self.train_set["labels"]
        )

        train_df, holdout_df, train_labels, holdout_labels, train_weights, holdout_weights = train_test_split(
            train_df,
            train_labels,
            train_weights,
            test_size=0.5,
            shuffle=True,
            stratify=train_labels
        )


        # Calculate the sum of weights for signal and background in the training and validation sets
        train_signal_weights = train_weights[train_labels == 1].sum()
        train_background_weights = train_weights[train_labels == 0].sum()
        valid_signal_weights = valid_weights[valid_labels == 1].sum()
        valid_background_weights = valid_weights[valid_labels == 0].sum()
        holdout_signal_weights = holdout_weights[holdout_labels == 1].sum()
        holdout_background_weights = holdout_weights[holdout_labels == 0].sum()

        # Balance the sum of weights for signal and background in the training and validation sets
        train_weights[train_labels == 1] *= signal_weights / train_signal_weights
        train_weights[train_labels == 0] *= background_weights / train_background_weights
        valid_weights[valid_labels == 1] *= signal_weights / valid_signal_weights
        valid_weights[valid_labels == 0] *= background_weights / valid_background_weights
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

        self.eval_set = [(self.train_set['data'], self.train_set['labels']), (valid_df.to_numpy(), valid_labels)]

        self.holdout = {
                "data": holdout_df,
                "labels": holdout_labels,
                "weights": holdout_weights
            }

        self.validation_sets = []
        for i in range(10):
            # Loop 10 times to generate 10 validation sets
            tes = round(np.random.uniform(0.9, 1.10), 2)
            # apply systematics
            valid_df_temp = valid_df.copy()
            valid_df_temp["weights"] = valid_weights
            valid_df_temp["labels"] = valid_labels

            valid_with_systematics_temp = self.systematics(
                data=valid_df_temp,
                tes=tes
            ).data
            # valid_with_systematics_temp = postprocess(valid_df_temp)

            valid_labels_temp = valid_with_systematics_temp.pop('labels')
            valid_weights_temp = valid_with_systematics_temp.pop('weights')
            valid_with_systematics = valid_with_systematics_temp.copy()

            self.validation_sets.append({
                "data": valid_with_systematics,
                "labels": valid_labels_temp,
                "weights": valid_weights_temp,
                "settings": self.train_set["settings"],
                "tes": tes
            })
            del valid_with_systematics_temp
            del valid_df_temp

        train_signal_weights = train_weights[train_labels == 1].sum()
        train_background_weights = train_weights[train_labels == 0].sum()
        valid_signal_weights = valid_weights[valid_labels == 1].sum()
        valid_background_weights = valid_weights[valid_labels == 0].sum()
        holdout_signal_weights = holdout_weights[holdout_labels == 1].sum()
        holdout_background_weights = holdout_weights[holdout_labels == 0].sum()

        print(f"[*] --- original signal: {signal_weights} --- original background: {background_weights}")
        print(f"[*] --- train signal: {train_signal_weights} --- train background: {train_background_weights}")
        print(f"[*] --- valid signal: {valid_signal_weights} --- valid background: {valid_background_weights}")
        print(f"[*] --- holdout signal: {holdout_signal_weights} --- holdout background: {holdout_background_weights}")

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
        # self.train_set['predictions'] = (self.train_set['data'], self.best_theta)

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

    def _predict(self, X, theta):
        Y_predict = self._return_score(X)
        predictions = (Y_predict > theta).astype(int)
        return predictions
    
    def _predict_holdout(self):
        print("[*] --- Predicting Holdout set")
        X_holdout = self.holdout['data']
        X_holdout_sc = self.scaler.transform(X_holdout)
        self.holdout['score'] = self._return_score(X_holdout_sc)
        print("[*] --- Predicting Holdout set done")
        print("[*] --- score = ", self.holdout['score'])
        # plt.hist(self.holdout['score'], bins=30)
        # plt.show()



    def mu_hat_calc(self):

        self.holdout['data'] = self.scaler.transform(self.holdout['data'])
        Y_hat_holdout = self._predict(self.holdout['data'], self.best_theta)  
        Y_holdout = self.holdout['labels']
        weights_holdout = self.holdout['weights']

        # compute gamma_roi
        weights_holdout_signal = weights_holdout[Y_holdout == 1]
        weights_holdout_bkg = weights_holdout[Y_holdout == 0]

        Y_hat_holdout_signal = Y_hat_holdout[Y_holdout == 1]
        Y_hat_holdout_bkg = Y_hat_holdout[Y_holdout == 0]

        self.gamma_roi = (weights_holdout_signal[Y_hat_holdout_signal == 1]).sum()

        # compute beta_roi
        self.beta_roi = (weights_holdout_bkg[Y_hat_holdout_bkg == 1]).sum()
        if self.gamma_roi == 0:
            self.gamma_roi = EPSILON


    def amsasimov_x(self, s, b):
        '''
        This function calculates the Asimov crossection significance for a given number of signal and background events.
        Parameters: s (float) - number of signal events

        Returns:    float - Asimov crossection significance
        '''

        if b <= 0 or s <= 0:
            return 0
        try:
            return s/sqrt(s+b)
        except ValueError:
            print(1+float(s)/b)
            print(2*((s+b)*log(1+float(s)/b)-s))
        # return s/sqrt(s+b)

    def del_mu_stat(self, s, b):
        '''
        This function calculates the statistical uncertainty on the signal strength.
        Parameters: s (float) - number of signal events
                    b (float) - number of background events

        Returns:    float - statistical uncertainty on the signal strength

        '''
        return (np.sqrt(s + b)/s)

    def get_meta_validation_set(self):

        meta_validation_data = []
        meta_validation_labels = []
        meta_validation_weights = []

        for valid_set in self.validation_sets:
            meta_validation_data.append(valid_set['data'])
            meta_validation_labels = np.concatenate((meta_validation_labels, valid_set['labels']))
            meta_validation_weights = np.concatenate((meta_validation_weights, valid_set['weights']))

        return {
            'data': pd.concat(meta_validation_data),
            'labels': meta_validation_labels,
            'weights': meta_validation_weights
        }

    # def _choose_theta_new(self):

    #     self.holdout['score']


    #     hold_out_hist, hold_out_bins = np.histogram(self.holdout['score'],
    #         bins=30, density=False, weights=self.holdout['weights'])
        
    #     min_bin = np.argmin(hold_out_hist)

    #     theta = self.holdout['score'][self.holdout['score'] > 

    def _choose_theta(self):

        print("[*] Choose best theta")

        meta_validation_set = self.validation_set
        val_min = 1
        # Loop over theta candidates
        # try each theta on meta-validation set
        # choose best theta
        for theta in tqdm(self.theta_candidates):
            meta_validation_set_df_sc = self.scaler.transform(meta_validation_set["data"])
            meta_validation_set['score'] = self._return_score(meta_validation_set_df_sc)

            weights_valid = meta_validation_set["weights"].copy()
            valid_df = meta_validation_set["data"][meta_validation_set['score'] > theta]
            valid_array = valid_df['DER_deltar_lep_had']
            weights_valid = weights_valid[meta_validation_set['score'] > theta]  
            Y_hat_valid = meta_validation_set['score'][meta_validation_set['score'] > theta] 
            # Get predictions from trained model


            # get region of interest

            # predict probabilities for holdout
            holdout_val = self.holdout['data']['DER_deltar_lep_had']
            Y_hat_holdout = self.holdout['score']
            Y_holdout = self.holdout['labels']
            weights_holdout = self.holdout['weights']
            # compute gamma_roi

            weights_holdout = weights_holdout[Y_hat_holdout > theta]
            holdout_val = holdout_val[Y_hat_holdout > theta]

            Y_holdout = Y_holdout[Y_hat_holdout > theta]

            weights_holdout_signal = weights_holdout[Y_holdout == 1]
            weights_holdout_bkg = weights_holdout[Y_holdout == 0]
            bins = self.bins
            
            gamma_roi ,bins = np.histogram(holdout_val[Y_holdout == 1],
                        bins=bins, density=False, weights=weights_holdout_signal)
            
            beta_roi , bins = np.histogram(holdout_val[Y_holdout == 0],
                        bins=bins, density=False, weights=weights_holdout_bkg)
            

            
            hist_llr = self.calculate_NLL(weights_valid,valid_array,beta_roi,gamma_roi)

            val =  np.abs(self.mu_scan[np.argmin(hist_llr)] - 1)

            if val < val_min:
                print("val: ", val)
                print("gamma_roi: ", gamma_roi)
                print("beta_roi: ", beta_roi)
                print("theta: ", theta)
                print("Uncertainity", np.sqrt(gamma_roi + beta_roi)/gamma_roi)
                val_min = val
                self.best_theta = theta

        print(f"[*] --- best theta: {self.best_theta}")

    def _validate(self):
        for valid_set in self.validation_sets:
            valid_set_sc= self.scaler.transform(valid_set['data'])
            # valid_set['predictions'] = self._predict(valid_set_sc, self.best_theta)
            valid_set['score'] = self._return_score(valid_set_sc)

    
    
    def _compute_validation_result(self):
        print("[*] - Computing Validation result")
        self.validation_mu_hats = []

        self.validation_delta_mu_hats = []
        for valid_set in self.validation_sets:

            Y_hat_valid_set = valid_set['score']
            Y_valid_set = valid_set['labels']
            weights_valid_set = valid_set['weights']

            valid_set_df = valid_set['data']
            valid_set_array = valid_set_df['DER_deltar_lep_had']
            # compute gamma_roi

            weights = weights_valid_set[Y_hat_valid_set > self.best_theta]
            valid_set_array = valid_set_array[Y_hat_valid_set > self.best_theta]

            Y_valid_set = Y_valid_set[Y_hat_valid_set > self.best_theta]

            mu_hat, mu_p16, mu_p84 = self._compute_result(weights,valid_set_array)

            self.validation_mu_hats.append(mu_hat)

            # Compute delta mu hat (absolute value)
            delta_mu_hat = np.abs(valid_set["settings"]["ground_truth_mu"] - mu_hat)


            self.validation_delta_mu_hats.append(delta_mu_hat)


            print(f"[*] --- p16: {np.round(mu_p16, 4)} --- p84: {np.round(mu_p84, 4)} --- mu_hat: {np.round(mu_hat, 4)}")

        measured_p16 = np.percentile(self.validation_mu_hats, 16)
        measured_p84 = np.percentile(self.validation_mu_hats, 84)

        self.calibration = [measured_p16, measured_p84]


        print(f"[*] --- validation delta_mu_hat (avg): {np.round(np.mean(self.validation_delta_mu_hats), 4)}")
        del self.validation_sets

    def _save_model(self):

        self.model.save("../DANN_saved/model.keras")

        settings = {
            "best_theta": self.best_theta,
            "calibration": self.calibration,
            "theta_candidates": self.theta_candidates,
            "bins": self.bins,
            "mu_scan": self.mu_scan,
            "beta_roi": self.beta_roi,
            "gamma_roi": self.gamma_roi
        }

        with open("settings.json", "w") as f:
            json.dump(settings, f)

        pickle.dump(self.scaler, open("scaler.pkl", "wb"))

        print("[*] - Model saved")