import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso

from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor

import xgboost as xgb
import lightgbm as lgb

from utils import define_network
from variables import *
from stacked_model import StackingAveragedModelsKeras


data = pd.read_csv('input_ex.csv', index_col=0)
y = data['GHI']
X = data.drop(columns=['GHI'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

callback = EarlyStopping(monitor='val_mse', patience=15, restore_best_weights=True)
clf_nn = KerasRegressor(build_fn=define_network(X_train), verbose=0, epochs=150, batch_size=50, validation_split=0.2, callbacks=[callback])
clf_lasso = Lasso(alpha =0.00003, random_state=1, normalize=False, max_iter=3000, tol=1e-5, selection='random')

clf_xgb = xgb.XGBRegressor(**best_xgb_found)
clf_lgb = lgb.LGBMRegressor(**best_lgb_found)
clf_gb = GradientBoostingRegressor(**best_gb_found)

base_models = dict(xgb=clf_xgb, nn=clf_nn, lgb=clf_lgb, gb=clf_gb)
stacked = StackingAveragedModelsKeras(base_models, clf_lasso, 5)

stacked.fit(X_train, y_train)
predictions = stacked.predict(X_test.values)