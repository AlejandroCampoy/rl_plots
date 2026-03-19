import wandb
import pandas as pd

api = wandb.Api()

# Variables que forman el path al run (formato: entity/project/run_id)
ENTITY = "ai-uponor"
PROJECT = "ai-smatrix-learning"
RUN_ID = "c7rqx7hm"

run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

history = run.history(pandas=True)

history.to_csv(f"data/{ENTITY}_{PROJECT}_{RUN_ID}_history.csv", index=False)