# README

This model was created as part of a group project set-up by University of Liverpool. The project exists on https://github.com/jgasmi/AirQuality-ML/tree/main.

## XGBoostRegressor Model selection


    Boosting ensemble models like XGBoost, iteratively improves the errors while strengthening the weak learners (Lima, 2024), thereby enhancing accuracy. This scalable nature of tree boosting technique combined with a flexibility that comes with a vast range of configurable hyperparameters, makes XGBoost suitable for a wide range of Classification and Regression datasets. Besides the hyperparameters like learning_rate, n_estimators and max_depth, XGBoost Regression algorithm offers greater latitude in configurable hyperparameters with approximate split finding algo-rithm (max_bin), cache-aware access, build in regularization (alpha, lambda) and tree structure control hyperparameters (subsample, colsample_bytree) (Wade, 2020) to reduce overfitting and advance the model generalization.
After selecting the most relevant features, the dataset distributions were examined (Figure 1).:
![1761298398066](image/README/1761298398066.png)

Figure 1. Dataset Distribution

    XGBoost is prone to overfitting since it tries to catch the noises in datasets (Lima, 2024). Initially, the model is trained with default hyperparameters, which gives us a model score of 94.96% and RMSE of 121.1604. Observing the importance of features with SHAP (see Figure.7), we see that PM2.5 adds the most value to final predictions followed by NOx. 

![1761298428575](image/README/1761298428575.png)

Figure 2. XGBoost Feature Importances

    Next step was the cross-validation to find the best parameters for tuning the model using a combination of GridSearchCV and KFold techniques. Training the model with the best hyperparameters like max_depth, subsample, min_child_weight and reg_lambda to reduce overfitting, we get an optimal R2 score of 95.51% and RMSE reduced to 114.3364.

![1761298488863](image/README/1761298488863.png)

Figure 3. XGBoost actual vs predicted values


### References:

Lima Marinho, T., Nascimento, D.C. and Pimentel, B.A. (2024) ‘Optimization on se-lecting XGBoost hyperparameters using meta‐learning’, Expert systems, 41(9). Avail-able at: https://doi.org/10.1111/exsy.13611.

SHAP (no date) ‘An introduction to explainable AI with Shapley value’. Available: https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html. (Accessed: 13 October 2025)

Wade, C. (2020) Hands-on gradient boosting with XGBoost and scikit-learn perform accessible machine learning and extreme gradient boosting with Python / Corey Wade. 1st ed. Place of publication not identified: Packt Publishing.