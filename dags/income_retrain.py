"""Переобучение модели оттока.

train     под с образом сервиса (тот же код, что в проде): dvc pull датасета из S3, income.train, всё в MLflow
promoted  дальше идём, только если гейт в train.py отдал новой версии алиас champion
rollout   перезапуск income-service: новые поды загрузят нового champion из реестра

Образ для обучения берётся из переменной income_image, её выставляет CI при каждом деплое.
"""
from datetime import UTC, datetime

from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.sdk import CronTriggerTimetable, Param, dag, task
from kubernetes.client import models as k8s


@dag(
    # каждый день в 03:00 по Москве; пропущенные запуски не догоняем, в том числе при снятии с паузы
    schedule=CronTriggerTimetable("0 3 * * *", timezone="Europe/Moscow"),
    catchup=False,
    params={"C": Param(1.0, type="number", exclusiveMinimum=0, description="регуляризация LogisticRegression")},
    tags=["income"],
)
def income_retrain():
    train = KubernetesPodOperator(
        task_id="train",
        name="income-train",
        namespace="mlops",
        in_cluster=True,
        image="{{ var.value.income_image }}",
        image_pull_policy="IfNotPresent",
        # данных в образе нет: DVC берёт их из S3 по адресу изнутри кластера (remote cluster)
        cmds=["sh", "-c", "uv run --no-sync dvc pull -r cluster datasets/adult.csv && uv run --no-sync python -m income.train"],
        env_vars={"MLFLOW_TRACKING_URI": "http://mlflow.mlops:5000", "C": "{{ params.C }}"},
        env_from=[k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name="s3-credentials"))],
        do_xcom_push=True,
        get_logs=True,
        on_finish_action="delete_pod",
    )

    @task.short_circuit
    def promoted(result: dict) -> bool:
        print(f"версия {result['version']}, roc_auc {result['roc_auc']}, champion до этого {result['champion_before']}")
        return result["promoted"]

    @task
    def rollout():
        from kubernetes import client, config

        config.load_incluster_config()
        now = datetime.now(UTC).isoformat()
        client.AppsV1Api().patch_namespaced_deployment(
            "income-service", "default",
            {"spec": {"template": {"metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": now}}}}},
        )
        print("income-service перезапущен", now)

    promoted(train.output) >> rollout()


income_retrain()
