from datetime import datetime
from opt.teacher import OnGridMPC
from utils.utils import make_forecasts_from_csv  # ou make_forecasts_from_single_csv

teacher = OnGridMPC(params_path="data/mpc.json")
start_dt = datetime(2007, 1, 1, 0, 0)

forecasts, _times = make_forecasts_from_csv(
    pv_csv="data/pv_5min_train.csv",
    load_csv="data/load_5min_train.csv",
    start_dt=start_dt,
    params=teacher.params,
)

teacher.build(start_dt=start_dt, forecasts=forecasts, E_hat_kwh=5.0)
teacher.solve("gurobi", tee=False)
sol = teacher.extract_solution()
fast_sol = teacher.extract_solution_fast()

from utils.plot_results import plot_solution

fig, axes, df = plot_solution(sol, save_path="results/plot.png", e_nom_kwh=5.0)


a = 1

print(fast_sol)