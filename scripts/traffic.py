"""Поток запросов к сервису из строк датасета, чтобы на дашборде Grafana было что смотреть.

  uv run python scripts/traffic.py --url http://income.localhost --rps 5 --bad 0.05 --recs 0.3

Через Ingress запросы делятся между подами и переживают rollout. Через port-forward нет:
он привязан к одному поду и обрывается, когда этот под удаляют.
--bad   доля заведомо плохих запросов (tenure = -1), они дают 422 на графике.
--recs  доля запросов рекомендаций для случайного номера клиента из диапазона Online Retail 12346..18287:
        примерно четверть таких номеров магазину незнакома, им достаётся популярное. Остановить: Ctrl+C.
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://income.localhost")
parser.add_argument("--rps", type=float, default=5)
parser.add_argument("--bad", type=float, default=0.0)
parser.add_argument("--recs", type=float, default=0.0)
args = parser.parse_args()

df = pd.read_csv("dataset/adult.csv")
df[df == '?'] = np.nan
    
df = df.rename(columns={'education.num': 'education_num', 
                        'marital.status': 'marital_status', 
                        'capital.gain': 'capital_gain',
                        'capital.loss': 'capital_loss', 
                        'hours.per.week': 'hours_per_week', 
                        'native.country': 'native_country',
                    })
NUMERIC = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
CATEGORICAL = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 
               'race', 'sex', 'native_country']
df = df[NUMERIC + CATEGORICAL]
rows = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in df.to_dict("records")]

codes: dict[int, int] = {}
start = time.time()
while True:
    if random.random() < args.recs:
        req = urllib.request.Request(f"{args.url}/v1/recommend/{random.randint(12346, 18287)}")
    else:
        row = dict(random.choice(rows))
        if random.random() < args.bad:
            row["age"] = -1
        req = urllib.request.Request(f"{args.url}/v1/predict", data=json.dumps(row).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            code = resp.status
    except urllib.error.HTTPError as e:
        code = e.code
    except (urllib.error.URLError, OSError):
        code = 0  # сервис недоступен: под перезапускается или оборвался port-forward
    codes[code] = codes.get(code, 0) + 1
    total = sum(codes.values())
    if total % 50 == 0:
        print(f"{total} запросов за {time.time() - start:.0f} с, коды {codes}", flush=True)
    time.sleep(1 / args.rps)
