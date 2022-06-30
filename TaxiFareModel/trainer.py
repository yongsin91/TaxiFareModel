import mlflow
import joblib
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = "[SG] [Singapore] [yongsin91] taxifaremodel compiled"


    def set_pipeline(self,model):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            (str(model)[:-2], model)
        ])
        return pipe

    def run(self,model, params):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline(model)
        self.pipeline = RandomizedSearchCV(self.pipeline, params, scoring='neg_root_mean_squared_error')
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri("https://mlflow.lewagon.ai/")
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, "User", "yongsin91")

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

if __name__ == "__main__":
    # get data
    df = get_data()

    # clean data
    df = clean_data(df)

    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    model_list = [LinearRegression(),Lasso(),Ridge(), RandomForestRegressor()]
    svm_list = ['linear', 'poly', 'rbf', 'sigmoid']
    params= {
        # "RandomForestRegressor__max_depth":[1,2,3,4,5,6,7,8,9,10],
        # "RandomForestRegressor__min_samples_split":[2,3,4,5],
        "RandomForestRegressor__max_features":[None,"sqrt","log2",0.3,0.2,0.1,0.4,0.5],
        "RandomForestRegressor__min_impurity_decrease":[0.0,0.2,0.4,0.6,0.8,1.0],
        "RandomForestRegressor__n_estimators":[20,40,60,80,100,120,140,160,180,200],
        "RandomForestRegressor__min_samples_leaf":[1,2,3,4,5,6,7,8,9,10]
        # "RandomForestRegressor__min_weight_fraction_leaf":[0.0,0.1,0.2,0.3,0.4,0.5],
        # "RandomForestRegressor__max_leaf_nodes":[None,2,3,4,5,6,7,8,9,10]
    }

    """
    Do a initial run test on 10_000 samples to
    identify the better machine learning model
    """

    # for i in range(30):
    #     for svm in svm_list:

    #         # build pipeline
    #         trainer = Trainer(X_train,y_train)

    #         # train the pipeline
    #         trainer.run(SVR(kernel = svm))

    #         # evaluate the pipeline
    #         rmse = trainer.evaluate(X_val, y_val)
    #         trainer.mlflow_log_metric("rmse",rmse)
    #         trainer.mlflow_log_param("model",f"SVM - {str(svm)}")

        # # build pipeline
        # trainer = Trainer(X_train,y_train)

        # # train the pipeline
        # trainer.run(RandomForestRegressor(),params)

        # # evaluate the pipeline
        # trainer.mlflow_log_metric("rmse",trainer.pipeline.best_score_)
        # trainer.mlflow_log_param("best_params",f"{trainer.pipeline.best_params_}")
        # print(f'round {1+i}')

    # build pipeline
    trainer = Trainer(X_train,y_train)

    # train the pipeline
    trainer.run(RandomForestRegressor(),params)
