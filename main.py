import json
import os
import pandas as pd
from nixtlats import TimeGPT
from utils import create_train_val_test_split, evaluate_forecast, gap_check
import plotly.io as pio
pio.renderers.default = "browser"

timegpt = TimeGPT(token=os.environ['TIMEGPT_TOKEN'])

# timegpt.validate_token()
# df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/air_passengers.csv')
# df.head()
# fig = timegpt.plot(df, time_col='timestamp', target_col='value')
# fig.show()

data_path = 'data'
file_name = "MAC000100"
selected_customers = ["MAC000119", "MAC000121", "MAC000117", "MAC000110", "MAC000108"]
figs = {}
for file_name in selected_customers:
    df = pd.read_csv(f"{data_path}/{file_name}.csv", index_col='DateTime',
                        parse_dates=True, names=['DateTime', 'customer_id', 'usage'],
                        header=0)
    df['usage'] = pd.to_numeric(df['usage'], errors='coerce')
    df = df[['usage']]
    df_extended = gap_check(df=df, column_name='usage', freq='30T')
    df_extended_cleaned = df_extended.ffill().bfill()
    gap_check(df=df_extended_cleaned, column_name='usage', freq='30T')
    df = df_extended_cleaned
    df.index = pd.DatetimeIndex(df.index, freq='30T')
    df['DateTime'] = df.index
    print(f"[{file_name}]: entire data set: {df.describe()}")

    train, test = create_train_val_test_split(df)
    # print(f"[{file_name}]: known historical data length: {len(train)}, forecast horizon: {len(test)}")
    actual_horizon = 48
    print(f"[{file_name}]: known historical data length: {len(train)}, forecast horizon: {actual_horizon}")
    timegpt_fcst_df = timegpt.forecast(df=train, finetune_steps=50, h=actual_horizon, time_col='DateTime', target_col='usage', freq='30T')
    fig_name = f"{file_name}_timegpt_forecast"
    figs[fig_name] = timegpt.plot(train.iloc[-actual_horizon*28::], timegpt_fcst_df, time_col='DateTime', target_col='usage')
    figs[fig_name].show()
    figs[fig_name].savefig(f"results/{fig_name}.png")

    test_set_result = evaluate_forecast(test['usage'].iloc[:actual_horizon], timegpt_fcst_df['TimeGPT'])
    print(f"[{file_name}]: testing result rmse: {test_set_result['rmse']}, mape: {test_set_result['mape']:.3%}")
    test_set_result.pop('mape_raw')
    test_set_result.pop('rmse_raw')
    with open(f"results/{file_name}_timegpt_forecast_metrics.txt", "w") as f:
        json.dump(test_set_result, f)