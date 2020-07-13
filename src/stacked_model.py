import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import numpy as np
from copy import deepcopy, copy
from sklearn.base import TransformerMixin, RegressorMixin, BaseEstimator


class StackingAveragedModelsKeras(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    Description of StackingAveragedModelsKeras

    Creates a stacked average estimator capable of improving the overall accuracy.
    Automatically scales input of estimator that have the 'nn' as key in the input dictionary.

    """
    
    
    def __init__(self, base_models, meta_model, n_folds=5):
        """
        Description of __init__

        Args:
            self (undefined):
            base_models (undefined): dictionary of base models
            meta_model (undefined): array of meta models
            n_folds=5 (undefined): number of folds for cross validation

        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.scalerx = MinMaxScaler()
        self.scalery = MinMaxScaler()
   
   
    def fit(self, X, y):
        """
        Description of fit
        
        Fit model on input data

        Args:
            self (undefined):
            X (undefined): X training data
            y (undefined): y training data

        """
    
        self.base_models_ = {key:list() for key, clf in self.base_models.items()}
        self.meta_model_ = deepcopy(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        X_sc = self.scalerx.fit_transform(X)
        y_sc = self.scalery.fit_transform(y.values.reshape(-1,1))
        X = X.values
        y = y.values

        i = -1
        for key, model in self.base_models.items():
            i += 1
            print(f"training model {key}")
            for train_index, holdout_index in kfold.split(X, y):
                print(f"new split")
                instance = deepcopy(model)
                self.base_models_[key].append(instance)
                if key != 'nn':
                    instance.fit(X[train_index], y[train_index])
                    y_pred = instance.predict(X[holdout_index])
                else:
                    instance.fit(X_sc[train_index], y_sc[train_index])
                    y_pred = instance.predict(X_sc[holdout_index])
                    y_pred = self.scalery.inverse_transform(y_pred.reshape(-1,1)).ravel()
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   

    def predict(self, X):
        """
        Description of predict
        Predict for test data

        Args:
            self (undefined):
            X (undefined): X test data

        """
        all_pred = []
        for name, base_models in self.base_models_.items():
            single_predictions = []
            for model in base_models:
                if name == 'nn':
                    X_sc = self.scalerx.transform(X)
                    pred = model.predict(X_sc)
                    pred = self.scalery.inverse_transform(pred.reshape(-1,1)).ravel()
                else:
                    pred = model.predict(X)
                single_predictions.append(pred)
            col = np.column_stack(single_predictions).mean(axis=1)
            all_pred.append(col)
        meta_features = np.column_stack(all_pred)
        return self.meta_model_.predict(meta_features)
    
    
    def save_model(self, filename):
        """
        Description of save_model
        
        Save trained model

        Args:
            self (undefined):
            filename (undefined): filename of model

        """
        joblib.dump(self, filename)
    
    
    @staticmethod  
    def load_model(filename):
        """
        Description of load_model
        Load trained model

        Args:
            filename (undefined): filename of model

        """
        return joblib.load(filename)

