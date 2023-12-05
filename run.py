import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin

# Version_7 is the way to go.

# /opt/anaconda/bin/python3 run.py -c configs/cvae.yaml
# python3 run.py -c configs/dfc_vae.yaml
# pip install --upgrade protobuf==3.9.2

# THESE ARE THE HOLY HOLY GRAILS!
# /home/yitongt/.conda/envs/prot2face/bin/pip3
# /home/yitongt/.conda/envs/prot2face/bin/python3



parser = argparse.ArgumentParser(description="Generic runner for VAE models")
parser.add_argument(
    "--config",
    "-c",
    dest="filename",
    metavar="FILE",
    help="path to the config file",
    default="configs/dfc_vae.yaml",
)

args = parser.parse_args()
with open(args.filename, "r") as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tb_logger = TensorBoardLogger(
    save_dir=config["logging_params"]["save_dir"],
    name=config["model_params"]["name"],
)
print('we are after tb_logger!')

# For reproducibility
seed_everything(config["exp_params"]["manual_seed"], True)
print('we are after seeding!')

model = vae_models[config["model_params"]["name"]](**config["model_params"])
experiment = VAEXperiment(model, config["exp_params"])
data = VAEDataset(
    **config["data_params"], pin_memory=len(config["trainer_params"]["gpus"]) != 0
)
print('we are after data loading!')

data.setup()

print('we are after data setup')
runner = Trainer(
    logger=tb_logger,
    # devices=2,
    callbacks=[
        LearningRateMonitor(),
        ModelCheckpoint(
            save_top_k=2,
            dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
            monitor="val_loss",
            save_last=True,
        ),
    ],
    strategy=DDPPlugin(find_unused_parameters=False),
    **config["trainer_params"],
)

print('we are after trainer')

Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)
