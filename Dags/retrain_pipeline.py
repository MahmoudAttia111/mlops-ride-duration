from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

dag = DAG(
    dag_id="arabic_sentiment_retrain",
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5), "owner": "ml-team"},
)

def train_task(**ctx):
    import mlflow
    import subprocess
    with mlflow.start_run() as run:
        subprocess.run(["python", "-m", "prodml.train"], check=True)
        ctx["ti"].xcom_push(key="run_id", value=run.info.run_id)

def register_task(**ctx):
    import subprocess
    subprocess.run(["python", "-m", "prodml.register_model"], check=True)

t_train = PythonOperator(task_id="train", python_callable=train_task, dag=dag)
t_register = PythonOperator(task_id="register", python_callable=register_task, dag=dag)

t_train >> t_register
