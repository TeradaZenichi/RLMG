# main.py
import json
from train.train_base import train as train_il
from testing.test_base import test as test_il
from train.train_rl import train as train_rl
from testing.test_rl import test as test_rl

def select_model(model_name: str):
    model_name = model_name.lower()
    if model_name == "td3mlp":
        import models.td3mlp.model as m
        config_path = "models/td3mlp/model.json"
        return m.DeterministicTanhActor, m.MLPmodel, m.TwinQCritic, m.Hyperparameters, config_path
    else:
        raise ValueError(f"modelo desconhecido: {model_name}")

if __name__ == "__main__":
    model_name = "td3mlp"  # default
    actor_cls, model_cls, critics_cls, hp_cls, config_path = select_model(model_name) 

    name = "td3mlp"
    start_time_train = "2006-12-17 00:00:00"
    start_soc_train  = 0.5  # fraction of Emax

    start_time_test = "2007-12-17 00:00:00"
    start_soc_test = 0.5  # fraction of Emax

    # # Imitation learning approach
    res = train_il(
        model_arch=model_cls,
        hp_cls=hp_cls,
        start_time=start_time_train,
        start_soc=start_soc_train,
        config_path=config_path,
        model_name=model_name
    )
    
    # Test the solution obtained by imitation learning
    res = test_il(
        start_time=start_time_test,
        start_soc=start_soc_test,
        model_arch=model_cls,
        hp_cls=hp_cls,
        config_path=config_path,
        name=name
    )

    # fine tunning using reinforcement learning
    weights_path = "saves/td3mlp/best_il.pt"
    # weights_path = None
    res = train_rl(
        actor_arch=actor_cls,
        model_arch=model_cls,
        critics_arch = critics_cls,
        hp_cls=hp_cls,
        start_time=start_time_train,
        start_soc=start_soc_train,
        config_path=config_path,
        model_name=model_name,
        model_weights=weights_path
    )

    res_rl_test = test_rl(
        start_time=start_time_test,
        start_soc=start_soc_test,
        hp_cls=hp_cls,
        config_path=config_path,
        name="td3mlp",           # pasta onde o train salvou best.pt/last.pt
        ckpt_name="last.pt"      # ou "last.pt"
    )