# train.py (raiz do projeto)
from models.model1.train import train, TrainConfig

if __name__ == "__main__":
    cfg = TrainConfig(
        ckpt_dir="saves/model1",   # >>> salva em saves/model1/sac_final.pt
        model_json="models/model1/model.json"

        # você pode ajustar outros campos aqui, se quiser:
        # start_days=["2006-12-17","2006-12-18","2006-12-19","2006-12-20"],
        # dt_minutes=15, horizon_hours=24, total_env_steps=200_000, ...
    )
    train(cfg)
