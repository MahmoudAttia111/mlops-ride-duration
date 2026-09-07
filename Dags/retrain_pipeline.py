from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

dag = DAG(
    dag_id="ride_duration_retrain",
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5), "owner": "ml-team"},
)

def extract(**ctx):
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket("mlops-session2-iti")
    bucket.blob("rides/latest.parquet").download_to_filename("data/raw/rides.parquet")

def train(**ctx):
    import mlflow
    import subprocess
    with mlflow.start_run() as run:
        subprocess.run(["python", "-m", "src.train"], check=True)
        ctx["ti"].xcom_push(key="run_id", value=run.info.run_id)

def evaluate(**ctx):
    import mlflow
    run_id = ctx["ti"].xcom_pull(key="run_id", task_ids="train")
    client = mlflow.MlflowClient()
    candidate_mae = mlflow.get_run(run_id).data.metrics.get("mae")
    try:
        prod_version = client.get_model_version_by_alias("RideDurationModel", "production")
        prod_mae = mlflow.get_run(prod_version.run_id).data.metrics.get("mae")
    except Exception:
        prod_mae = float("inf")
    if candidate_mae < prod_mae:
        client.set_registered_model_alias("RideDurationModel", "production",
                                            mlflow.get_run(run_id).info.run_id)
        print(f"Promoted run {run_id}: MAE {candidate_mae:.3f} < {prod_mae:.3f}")
    else:
        print(f"Rejected: MAE {candidate_mae:.3f} >= production {prod_mae:.3f}")

t_extract = PythonOperator(task_id="extract", python_callable=extract, dag=dag)
t_train = PythonOperator(task_id="train", python_callable=train, dag=dag)
t_evaluate = PythonOperator(task_id="evaluate", python_callable=evaluate, dag=dag)
t_extract >> t_train >> t_evaluate