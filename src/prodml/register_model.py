import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

MODEL_NAME = "ArabicSentimentModel"

def register_latest_run():
    experiment = client.get_experiment_by_name("arabic-sentiment")
    runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
    run = runs[0]

    result = mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, "production", result.version)
    print(f"Registered version {result.version} as @production (run {run.info.run_id})")

if __name__ == "__main__":
    register_latest_run()
