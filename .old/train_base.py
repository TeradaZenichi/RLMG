from datetime import datetime, timedelta
from opt import Teacher
import pandas as pd
import json

import environment

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
    teacher.solve()
    actions = teacher.normalized_actions
    states = teacher.normalized_states
    
   

    a = 123  # breakpoint
    