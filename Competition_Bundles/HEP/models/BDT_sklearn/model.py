import os
from sys import path
import numpy as np
import pandas as pd
from math import sqrt, log
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
import matplotlib.pyplot as plt 
import pickle 
import json 



import mplhep as hep

hep.set_style("ATLAS")

# ------------------------------
# Absolute path to submission dir
# ------------------------------
submissions_dir = os.path.dirname(os.path.abspath(__file__)) 
path.append(submissions_dir)


from systematics import postprocess
# from bootstrap import bootstrap
# ------------------------------
# Constants
# ------------------------------
EPSILON = np.finfo(float).eps



# HPO = True
HPO = False




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

            test_sets:
                unlabelled test sets

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
        self.threshold_candidates = np.arange(0.5, 0.99, 0.01)
        self.threshold = 0.85
        self.scaler = StandardScaler()
        self.scaler_tes = StandardScaler()

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
        # self._choose_theta()
        self.mu_hat_calc()
        self._validate()
        self._compute_validation_result()
        self._theta_plot()
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
        Y_hat_test = self._predict(test_df, self.threshold)

        print("[*] - Computing Test result")
        weights_train = self.train_set["weights"].copy()
        weights_test = test_set["weights"].copy()

        print(f"[*] --- total weight test: {weights_test.sum()}") 
        print(f"[*] --- total weight train: {weights_train.sum()}")
        print(f"[*] --- total weight mu_cals_set: {self.mu_calc_set['weights'].sum()}")

        # get n_roi
        n_roi = (weights_test[Y_hat_test == 1]).sum()

        mu_hat = (n_roi - self.beta_roi)/self.gamma_roi

        sigma_mu_hat = np.sqrt(n_roi)/self.gamma_roi

        delta_mu_hat = 2*sigma_mu_hat

        mu_p16 = mu_hat - sigma_mu_hat
        mu_p84 = mu_hat + sigma_mu_hat

        print(f"[*] --- mu_hat: {mu_hat}")
        print(f"[*] --- delta_mu_hat: {delta_mu_hat}")
        print(f"[*] --- p16: {mu_p16}")
        print(f"[*] --- p84: {mu_p84}")

        return {
            "mu_hat": mu_hat.mean(),
            "delta_mu_hat": delta_mu_hat,
            "p16": mu_p16,
            "p84": mu_p84
        }


    if HPO == False: 
        def _init_model(self):
            print("[*] - Intialize Baseline Model (HGBC)")

            self.model = ensemble.HistGradientBoostingClassifier()

    if HPO == True: 
        def _init_model(self):
            print("[*] - Intialize Baseline Model (HPO for HGBC)")

            param_dist = {'max_depth': stats.randint(3, 12), 
                      'learning_rate': stats.uniform(0.1, 0.5)} 

            self.model = RandomizedSearchCV(estimator = ensemble.HistGradientBoostingClassifier(),
                        param_distributions = param_dist,
                        scoring='roc_auc',n_iter=10,cv=5)


        
    def _generate_validation_sets(self):
        print("[*] - Generating Validation sets")

        # Calculate the sum of weights for signal and background in the original dataset
        signal_weights = self.train_set["weights"][self.train_set["labels"] == 1].sum()
        background_weights = self.train_set["weights"][self.train_set["labels"] == 0].sum()

        # Split the data into training and validation sets while preserving the proportion of samples with respect to the target variable
        train_df, valid_df, train_labels, valid_labels, train_weights, valid_weights = train_test_split(
            self.train_set["data"], 
            self.train_set["labels"],
            self.train_set["weights"],
            test_size=0.05,
            stratify=self.train_set["labels"]
        )

        train_df, mu_calc_set_df, train_labels, mu_calc_set_labels, train_weights, mu_calc_set_weights = train_test_split(
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
        mu_calc_set_signal_weights = mu_calc_set_weights[mu_calc_set_labels == 1].sum()
        mu_calc_set_background_weights = mu_calc_set_weights[mu_calc_set_labels == 0].sum()

        # Balance the sum of weights for signal and background in the training and validation sets
        train_weights[train_labels == 1] *= signal_weights / train_signal_weights
        train_weights[train_labels == 0] *= background_weights / train_background_weights
        valid_weights[valid_labels == 1] *= signal_weights / valid_signal_weights
        valid_weights[valid_labels == 0] *= background_weights / valid_background_weights
        mu_calc_set_weights[mu_calc_set_labels == 1] *= signal_weights / mu_calc_set_signal_weights
        mu_calc_set_weights[mu_calc_set_labels == 0] *= background_weights / mu_calc_set_background_weights

        train_df = train_df.copy()
        train_df["weights"] = train_weights
        train_df["labels"] = train_labels
        train_df = postprocess(train_df)

        train_weights = train_df.pop('weights')
        train_labels = train_df.pop('labels')
        

        mu_calc_set_df = mu_calc_set_df.copy()
        mu_calc_set_df["weights"] = mu_calc_set_weights
        mu_calc_set_df["labels"] = mu_calc_set_labels
        mu_calc_set_df = postprocess(mu_calc_set_df)

        mu_calc_set_weights = mu_calc_set_df.pop('weights')
        mu_calc_set_labels = mu_calc_set_df.pop('labels')




        self.train_df = train_df

        self.train_set = {
            "data": train_df,
            "labels": train_labels,
            "weights": train_weights,
            "settings": self.train_set["settings"]
        }

        self.eval_set = [(self.train_set['data'], self.train_set['labels']), (valid_df.to_numpy(), valid_labels)]

        self.mu_calc_set = {
                "data": mu_calc_set_df,
                "labels": mu_calc_set_labels,
                "weights": mu_calc_set_weights
            }
        

        # print(self.mu_calc_set['data'])


        self.validation_sets = []
        # Loop 10 times to generate 10 validation sets
        for i in range(0, 20):
            tes = round(np.random.uniform(0.9, 1.10), 2)
            # tes = 1.0
            # apply systematics
            valid_df_temp = valid_df.copy()
            valid_df_temp["weights"] = valid_weights
            valid_df_temp["labels"] = valid_labels

            valid_with_systematics_temp = self.systematics(
                data=valid_df_temp,
                tes=tes
            ).data

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
        mu_calc_set_signal_weights = mu_calc_set_weights[mu_calc_set_labels == 1].sum()
        mu_calc_set_background_weights = mu_calc_set_weights[mu_calc_set_labels == 0].sum()

        print(f"[*] --- original signal: {signal_weights} --- original background: {background_weights}")
        print(f"[*] --- train signal: {train_signal_weights} --- train background: {train_background_weights}")
        print(f"[*] --- valid signal: {valid_signal_weights} --- valid background: {valid_background_weights}")
        print(f"[*] --- mu_calc_set signal: {mu_calc_set_signal_weights} --- mu_calc_set background: {mu_calc_set_background_weights}")

    def _train(self):


        weights_train = self.train_set["weights"].copy()
        train_labels = self.train_set["labels"].copy()
        train_data = self.train_set["data"].copy()
        class_weights_train = (weights_train[train_labels == 0].sum(), weights_train[train_labels == 1].sum())

        for i in range(len(class_weights_train)):  # loop on B then S target
            # training dataset: equalize number of background and signal
            weights_train[train_labels == i] *= max(class_weights_train) / class_weights_train[i]
            # test dataset : increase test weight to compensate for sampling

        print("[*] --- Training Model")
        train_data = self.scaler.fit_transform(train_data)

        print("[*] --- shape of train tes data", train_data.shape)

        self._fit(train_data, train_labels, weights_train)

        print("[*] --- Predicting Train set")
        self.train_set['predictions'] = (self.train_set['data'], self.threshold)

        self.train_set['score'] = self._return_score(self.train_set['data'])

        auc_train = roc_auc_score(
            y_true=self.train_set['labels'],
            y_score=self.train_set['score'],
            sample_weight=self.train_set['weights']
        )
        print(f"[*] --- AUC train : {auc_train}")


    if HPO == True: 
        def _fit(self, X, y, w):
            print("[*] --- Fitting Model for RandomizedSearchCV")
            self.model.fit(X, y, sample_weight=w)
            dfsearch=pd.DataFrame.from_dict(self.model.cv_results_)
            #Printing of the results in a human-readable file to be able to think about HPO later on 
            path_to_csv = os.path.join("C:/","Users", "remyl", "fair-universe", "Competition_Bundles","HEP","models","BDT_sklearn","cv_results.csv")
            dfsearch.to_csv(path_to_csv, sep="\t")
            best_params = self.model.best_params_
            print("[*] --- Best parameters:", best_params)
            print("[*] --- Best estimator:", self.model.best_estimator_)
            print("[*] --- Fitting Model with best paramaters")
            # self.model.refit(True)


    if HPO == False: 
        def _fit(self, X, y, w): 
            print("[*] --- Fitting Model")
            self.model.fit(X, y, sample_weight=w)

    def _return_score(self, X):
        y_pred_skgb = self.model.predict_proba(X)[:,1]
        y_predict = y_pred_skgb.ravel()
        return y_predict

    def _predict(self, X, theta):
        Y_predict = self._return_score(X)
        predictions = np.where(Y_predict > theta, 1, 0)  
        return predictions


    def mu_hat_calc(self):

        X_holdout = self.mu_calc_set['data'].copy()
        X_holdout['weights'] = self.mu_calc_set['weights'].copy()
        X_holdout['labels'] = self.mu_calc_set['labels'].copy()

        holdout_post = self.systematics(
            data = X_holdout.copy(), 
            tes = 1.0
        ).data


        label_holdout = holdout_post.pop('labels')
        weights_holdout = holdout_post.pop('weights')
        X_holdout_sc = self.scaler.transform(holdout_post)

        holdout_score = self._return_score(X_holdout_sc)

        weights_holdout_signal= weights_holdout[label_holdout == 1]
        weights_holdout_bkg = weights_holdout[label_holdout == 0]

        score_holdout_signal = holdout_score[label_holdout == 1]
        score_holdout_bkg = holdout_score[label_holdout == 0]

        self.gamma_roi = (weights_holdout_signal[score_holdout_signal > self.threshold]).sum()
        if self.gamma_roi == 0:
            self.gamma_roi = EPSILON

        self.beta_roi = (weights_holdout_bkg[score_holdout_bkg > self.threshold]).sum()

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

    def _choose_theta(self):

        print("[*] Choose best theta")

        meta_validation_set = self.get_meta_validation_set()
        theta_sigma_squared = []

        # Loop over theta candidates
        # try each theta on meta-validation set
        # choose best theta
        for theta in self.threshold_candidates:
            meta_validation_set_df = self.scaler.transform(meta_validation_set["data"])    
            # Get predictions from trained model
            Y_hat_valid = self._predict(meta_validation_set_df, theta)
            Y_valid = meta_validation_set["labels"]

            weights_valid = meta_validation_set["weights"].copy()

            # get region of interest
            nu_roi = (weights_valid[Y_hat_valid == 1]).sum()/10

            weights_valid_signal = weights_valid[Y_valid == 1]  
            weights_valid_bkg = weights_valid[Y_valid == 0]

            Y_hat_valid_signal = Y_hat_valid[Y_valid == 1]  
            Y_hat_valid_bkg = Y_hat_valid[Y_valid == 0] 

            # compute gamma_roi
            gamma_roi = (weights_valid_signal[Y_hat_valid_signal == 1]).sum()/10

            # compute beta_roi
            beta_roi = (weights_valid_bkg[Y_hat_valid_bkg == 1]).sum()/10

            # Compute sigma squared mu hat
            sigma_squared_mu_hat = nu_roi/np.square(gamma_roi)

            # get N_ROI from predictions
            theta_sigma_squared.append(sigma_squared_mu_hat)

            print(f"\n[*] --- theta: {theta}--- nu_roi: {nu_roi} --- beta_roi: {beta_roi} --- gamma_roi: {gamma_roi} --- sigma squared: {sigma_squared_mu_hat}")

        # Choose theta with min sigma squared
        try:
            index_of_least_sigma_squared = np.nanargmin(theta_sigma_squared)
        except:
            print("[!] - WARNING! All sigma squared are nan")
            index_of_least_sigma_squared = np.argmin(theta_sigma_squared)

        self.threshold = self.threshold_candidates[index_of_least_sigma_squared]
        print(f"[*] --- Best theta : {self.threshold}")

    def _validate(self):
        for valid_set in self.validation_sets:
            valid_set['data'] = self.scaler.transform(valid_set['data'])
            valid_set['predictions'] = self._predict(valid_set['data'], self.threshold)
            valid_set['score'] = self._return_score(valid_set['data'])

    def _compute_validation_result(self):
        print("[*] - Computing Validation result")

        self.validation_delta_mu_hats = []
        for valid_set in self.validation_sets:
            Y_hat_train = self.train_set["predictions"]
            Y_train = self.train_set["labels"]
            Y_hat_valid = valid_set["predictions"]
            Y_valid = valid_set["labels"]
            Score_train = self.train_set["score"]
            Score_valid = valid_set["score"]

            auc_valid = roc_auc_score(y_true=valid_set["labels"], y_score=Score_valid,sample_weight=valid_set['weights'])
            print(f"\n[*] --- AUC validation : {auc_valid} --- tes : {valid_set['tes']}")

            # print(f"[*] --- PRI_had_pt : {valid_set['had_pt']}")
            # del Score_valid
            weights_train = self.train_set["weights"].copy()
            weights_valid = valid_set["weights"].copy()

            print(f'[*] --- total weights train: {weights_train.sum()}')
            print(f'[*] --- total weights valid: {weights_valid.sum()}')

            signal_valid = weights_valid[Y_valid == 1]
            background_valid = weights_valid[Y_valid == 0]

            Y_hat_valid_signal = Y_hat_valid[Y_valid == 1]
            Y_hat_valid_bkg = Y_hat_valid[Y_valid == 0]

            signal = signal_valid[Y_hat_valid_signal == 1].sum()
            background = background_valid[Y_hat_valid_bkg == 1].sum()

            significance = self.amsasimov_x(signal,background)
            print(f"[*] --- Significance : {significance}")

            delta_mu_stat = self.del_mu_stat(signal,background)
            print(f"[*] --- delta_mu_stat : {delta_mu_stat}")

            # get n_roi
            n_roi = (weights_valid[Y_hat_valid == 1]).sum()

            mu_hat = ((n_roi - self.beta_roi)/self.gamma_roi)
            # get region of interest
            nu_roi = self.beta_roi + self.gamma_roi

            print(f'[*] --- number of events in roi validation {n_roi}')
            print(f'[*] --- number of events in roi train {nu_roi}')

            gamma_roi = self.gamma_roi

            # compute beta_roi
            beta_roi = self.beta_roi
            if gamma_roi == 0:
                gamma_roi = EPSILON

            # Compute mu_hat

            # Compute delta mu hat (absolute value)
            delta_mu_hat = np.abs(valid_set["settings"]["ground_truth_mu"] - mu_hat)

            self.validation_delta_mu_hats.append(delta_mu_hat)

            print(f"[*] --- nu_roi: {nu_roi} --- n_roi: {n_roi} --- beta_roi: {beta_roi} --- gamma_roi: {gamma_roi}")

            print(f"[*] --- mu: {np.round(valid_set['settings']['ground_truth_mu'], 4)} --- mu_hat: {np.round(mu_hat, 4)} --- delta_mu_hat: {np.round(delta_mu_hat, 4)}")

        print(f"[*] --- validation delta_mu_hat (avg): {np.round(np.mean(self.validation_delta_mu_hats), 4)}")

        del self.validation_sets



    def nominal(self, theta):
        """
        Params: theta (the systematics) 

        Functionality: determine nominal s and b, ie the signal rate and the background rate in
                       the region of interest for different thetas (ie for different value for tes)

        Returns: s, b
        """

        X_holdout = self.mu_calc_set['data'].copy()
        # print(X_holdout)
        X_holdout['weights'] = self.mu_calc_set['weights'].copy()
        X_holdout['labels'] = self.mu_calc_set['labels'].copy()

        holdout_post = self.systematics(
            data = X_holdout.copy(), 
            tes = theta
        ).data


        label_holdout = holdout_post.pop('labels')
        weights_holdout = holdout_post.pop('weights')
        X_holdout_sc = self.scaler.transform(holdout_post)

        holdout_score = self._return_score(X_holdout_sc)

        weights_holdout_signal= weights_holdout[label_holdout == 1]
        weights_holdout_bkg = weights_holdout[label_holdout == 0]

        score_holdout_signal = holdout_score[label_holdout == 1]
        score_holdout_bkg = holdout_score[label_holdout == 0]

        s = (weights_holdout_signal[score_holdout_signal > self.threshold]).sum()
        if s == 0:
            s = EPSILON

        b = (weights_holdout_bkg[score_holdout_bkg > self.threshold]).sum()

        return s, b

        # X_mu_calc = self.mu_calc_set['data'].copy()
        # # print(X_mu_calc)
        # X_mu_calc['weights'] = self.mu_calc_set['weights'].copy()
        # X_mu_calc['labels'] = self.mu_calc_set['labels'].copy()

        # mu_calc_syst = self.systematics(
        #     data=X_mu_calc.copy(),
        #     tes=theta
        # ).data


        # label_mu_calc = mu_calc_syst.pop('labels')
        # weights_mu_calc = mu_calc_syst.pop('weights')

        # X_mu_calc_sc = self.scaler.transform(mu_calc_syst)
        # mu_calc_val = self._return_score(X_mu_calc_sc)

        # weights_mu_calc_signal = weights_mu_calc[label_mu_calc == 1]
        # weights_mu_calc_bkg = weights_mu_calc[label_mu_calc == 0]


        # mu_calc_val_signal = mu_calc_val[label_mu_calc ==1]
        # mu_calc_val_bkg = mu_calc_val[label_mu_calc ==0]

        # s = (weights_mu_calc_signal[mu_calc_val>self.threshold]).sum()  
        # b = (weights_mu_calc_bkg[mu_calc_val_bkg>self.threshold]).sum()
        # if s == 0:
        #     s = EPSILON


        # return s, b
    


    def _theta_plot(self):
        """
        Params: None

        Functionality: Save the plots in the same file as the model serialization (see _save_model)

        Returns: None
        """

        print("[*] Saving the plots")

        
        theta_list = np.linspace(0.9,1.1,10)
        s_list = []
        b_list = []
        
        for theta in theta_list:
            s , b = self.nominal(theta)
            s_list.append(s)
            b_list.append(b)
            # print(f"[*] --- s: {s}")
            # print(f"[*] --- b: {b}")


        fig_s = plt.figure()
        plt.plot(theta_list, s_list, 'b.', label = 's')
        plt.xlabel('theta')
        plt.ylabel('events')
        plt.legend(loc = 'lower right')
        hep.atlas.text(loc=1, text = " ")

        # plot file location on Atlas1 (same as local, but I can use linux functionalities for paths)
        save_path_s = os.path.join(submissions_dir, "Plots and serialization/")
        plot_file_s = os.path.join(save_path_s, "HGBC_s.png")

        plt.savefig(plot_file_s)
        plt.close(fig_s) # So the figure is not diplayed 
        



        fig_b = plt.figure()
        plt.plot(theta_list, b_list, 'b.', label = 'b')
        plt.xlabel('theta')
        plt.ylabel('events')
        plt.legend(loc = 'lower right')
        hep.atlas.text(loc=1, text = " ")

        # plot file location on Atlas1 (same as local, but I can use linux functionalities for paths)
        save_path_b = os.path.join(submissions_dir, "Plots and serialization/")
        plot_file_b = os.path.join(save_path_b, "HGBC_b.png")

        plt.savefig(plot_file_b)
        plt.close(fig_b) # So the figure is not diplayed 

        del self.mu_calc_set


    def _save_model(self):


        save_dir= os.path.join(submissions_dir, "Plots and serialization/")
        model_path = os.path.join(save_dir, "model.pkl")
        settings_path = os.path.join(save_dir, "settings.pkl")
        scaler_path = os.path.join(save_dir, 'scaler.pkl')

        print("[*] Saving Model")
        print(f"[*] --- model path: {model_path}")
        print(f"[*] --- settings path: {settings_path}")
        print(f"[*] --- scaler path: {scaler_path}")


        settings = {
            "threshold": self.threshold,
            "beta_roi": self.beta_roi,
            "gamma_roi": self.gamma_roi
        }


        pickle.dump(self.model, open(model_path, "wb"))

        pickle.dump(settings, open(settings_path, "wb"))

        pickle.dump(self.scaler, open(scaler_path, "wb"))

        print("[*] - Model saved")