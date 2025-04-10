import ray
import os
import numpy as np
from ray import serve
import os

try:
    MLFLOW_SERVICE=os.environ["MLFLOW_SERVICE"]
except:
    MLFLOW_SERVICE="localhost:8080"

ray.init()

@ray.remote
def data_preprocessing():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/MayurSatav/Wine-Quality-Prediction/refs/heads/master/winequality-red.csv"
    )

    data = pd.read_csv(csv_url, sep=",")

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    return (train_x, test_x, train_y, test_y)

@ray.remote
def train(train_x, test_x, train_y, test_y):
    import mlflow
    import pandas as pd
    import logging
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import ElasticNet
    np.random.seed(40)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def eval_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    MLFLOW_TRACKING_URI = "http://" + MLFLOW_SERVICE # The cluster internal mlflow service
    MLFLOW_EXPERIMENT_NAME = "score-mlflow-wine"

    alpha = 0.5
    l1_ratio = 0.5

    logger.info(f"Using MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    logger.info(f"Using MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)

        logger.info("Fitting model...")

        lr.fit(train_x, train_y)

        logger.info("Finished fitting")

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        logger.info("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        logger.info("  RMSE: %s" % rmse)
        logger.info("  MAE: %s" % mae)
        logger.info("  R2: %s" % r2)

        logger.info("Logging parameters to MLflow")
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        logger.info("Logging trained model")
        mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")

train_x, test_x, train_y, test_y = ray.get(data_preprocessing.remote())


ray.get(train.remote(np.array(train_x), np.array(test_x), np.array(train_y), np.array(test_y)))


serve_options = { "host" : "0.0.0.0"}
serve.start(http_options=serve_options)

@serve.deployment()
class Model:
    import starlette.requests
    def __init__(self):
        import mlflow
        from mlflow import MlflowClient
        MLFLOW_TRACKING_URI = "http://" + MLFLOW_SERVICE
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        LATEST_VERSION = client.get_latest_versions("ElasticnetWineModel")[0].version
        self.model = mlflow.pyfunc.load_model("models:/ElasticnetWineModel/" + LATEST_VERSION)
    async def __call__(self, request: starlette.requests.Request):
        data = await request.json()
        return self.predict(data)
    def predict(self, data):
        model_output = self.model.predict(data)
        return model_output

serve.run(Model.bind(), route_prefix="/elasticnetwine/predict")
