from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from data_loader.download_data import download_and_save_data
from data_loader.feature_engineering import process_and_save_all
from services.model_training import train_and_save_model
from services.model_evaluation import evaluate_model_performance

# Get current date for dynamic start date
start_date = datetime.now()

default_args = {
    'owner': 'portfolio_optimizer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def conditional_train_model():
    """Train the model only on 1st day of even months."""
    day = int(datetime.now().day)
    month = int(datetime.now().month)
    if day == 1 and month % 2 == 0:
        print("First of even month. Training model...")
        train_and_save_model()
    else:
        print("Not first of even month. Skipping training.")

def conditional_evaluate_model():
    """Evaluate the model only on 1st day of even months."""
    day = int(datetime.now().day)
    month = int(datetime.now().month)
    if day == 1 and month % 2 == 0:
        print("First of even month. Evaluating model...")
        evaluate_model_performance()
    else:
        print("Not first of even month. Skipping evaluation.")

with DAG(
    dag_id='portfolio_optimizer_dag',
    default_args=default_args,
    description='Daily load data, even month train and evaluate model (no DVC)',
    schedule_interval='@daily',
    start_date=start_date,
    catchup=False,
    tags=['portfolio', 'optimizer'],
) as dag:

    # Step 1: Download raw data
    download_data_task = PythonOperator(
        task_id='download_data_task',
        python_callable=download_and_save_data,
    )

    # Step 2: Feature engineering
    feature_engineering_task = PythonOperator(
        task_id='feature_engineering_task',
        python_callable=process_and_save_all,
    )

    # Step 3: Conditional model training
    train_model_task = PythonOperator(
        task_id='train_model_task',
        python_callable=conditional_train_model,
    )

    # Step 4: Conditional model evaluation
    evaluate_model_task = PythonOperator(
        task_id='evaluate_model_task',
        python_callable=conditional_evaluate_model,
    )

    # Define task order
    download_data_task >> feature_engineering_task >> train_model_task >> evaluate_model_task
