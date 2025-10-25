from datetime import datetime, timedelta
from opt import Teacher
import pandas as pd
import json

# Adjust project root for imports
import sys
import os
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)


START_TIME = "2006-12-17 00:00:00"
START_SoC = 0.5  # kWh

if __name__ == "__main__":
    params  = json.load(open("data/parameters.json"))
    pv      = pd.read_csv("data/pv_5min_train.csv", index_col=0, parse_dates=True)
    load    = pd.read_csv("data/load_5min_train.csv", index_col=0, parse_dates=True)

    # convert START_TIME to datetime
    start_time = datetime.strptime(START_TIME, "%Y-%m-%d %H:%M:%S")
    teacher = Teacher.Teacher(params, pv, load)
    teacher.build(start_time, START_SoC)
    results = teacher.solve()
    df = teacher.results2dataframe()
    df.to_csv("teacher_results.csv", index=False)


    from utils.plot_results import plot_solution_from_df
    plot_solution_from_df(df, params["BESS"]["Emax"], title="Teacher Solution", save_path="teacher_solution.pdf")
    
