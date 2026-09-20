import pandas as pd

cols = ['Name', 'Requests/s', 'Median Response Time', '95%', 'Max Response Time', 'Failure Count']
frames = []

for name in ['10', '50', '100']:
    df = pd.read_csv(f'locust_results/run{name}_stats.csv')

    rows = df[cols].copy()

    rows['Users'] = name

    frames.append(rows)

df_final = pd.concat(frames, ignore_index=True)
df_final.to_csv('locust_results/run_final.csv', index=False)