from models.td3mlp.model import GaussianTanhActor, TwinQCritic, Hyperparameters
from train.train_rl import train_rl
import json

file = "models/td3mlp/model.json"

if __name__ == "__main__":
    train_rl(
        actor_arch=GaussianTanhActor,
        critic_arch=TwinQCritic,
        hp_cls=Hyperparameters,
        start_time="2006-12-17 00:00:00",
        start_soc=0.5,
        config_path=file,
        model_name="td3mlp_energy"
    )

    a = 1