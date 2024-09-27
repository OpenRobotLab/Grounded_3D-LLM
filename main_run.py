import logging
import os
from hashlib import md5
from uuid import uuid4
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from trainer.trainer import ModelingGrounded3DLLM
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import (
    flatten_dict,
    load_checkpoint_with_missing_or_exsessive_keys,
)
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
import MinkowskiEngine as ME

class RegularCheckpointing(pl.Callback):
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        general = pl_module.config.general
        trainer.save_checkpoint(f"{general.save_dir}/last-epoch.ckpt")
        if trainer.global_rank == 0:
            print("Checkpoint created")

def get_parameters(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # parsing input parameters
    seed_everything(cfg.general.seed)

    # getting basic configuration
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    loggers = []

    cfg.general.experiment_id = "0" # str(Repo("./").commit())[:8]
    params = flatten_dict(OmegaConf.to_container(cfg, resolve=True))

    # create unique id for experiments that are run locally
    unique_id = "_" + str(uuid4())[:4]
    cfg.general.version = md5(str(params).encode("utf-8")).hexdigest()[:8] + unique_id

    if not os.path.exists(cfg.general.save_dir):
        os.makedirs(cfg.general.save_dir)
    else:
        if os.path.exists(f"{cfg.general.save_dir}/last-epoch.ckpt"):
            cfg["trainer"][
                "resume_from_checkpoint"
            ] = f"{cfg.general.save_dir}/last-epoch.ckpt"
            # if cfg.general.train_mode is False:
            print(f'Load weights from: {f"{cfg.general.save_dir}/last-epoch.ckpt"}')
            cfg.general.checkpoint = f"{cfg.general.save_dir}/last-epoch.ckpt"
        else:
            print(f'Note that *No* checkpoint is found.')

    for log in cfg.logging:
        print(log)
        loggers.append(hydra.utils.instantiate(log))
        loggers[-1].log_hyperparams(
            flatten_dict(OmegaConf.to_container(cfg, resolve=True))
        )

    model = ModelingGrounded3DLLM(cfg)
    if cfg.general.gpus > 1:
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)

    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, loggers


@hydra.main(
    config_path="conf", config_name="config_base.yaml"
)
def train(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    callbacks = []
    for cb in cfg.callbacks:
        callbacks.append(hydra.utils.instantiate(cb))

    callbacks.append(RegularCheckpointing())

    runner = Trainer(
        logger=loggers,
        gpus=cfg.general.gpus,
        accelerator='gpu' if cfg.general.gpus > 1 else None,
        strategy="ddp" if cfg.general.gpus > 1 else None,
        callbacks=callbacks,
        weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer,
    )
    runner.fit(model)


@hydra.main(
    config_path="conf", config_name="config_base.yaml"
)
def test(cfg: DictConfig):
    # because hydra wants to change dir for some reason
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        gpus=cfg.general.gpus,
        logger=loggers,
        accelerator='gpu' if cfg.general.gpus > 1 else None,
        strategy="ddp" if cfg.general.gpus > 1 else None,
        weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer,
    )
    runner.test(model)


@hydra.main(
    config_path="conf", config_name="config_base.yaml"
)
def main(cfg: DictConfig):
    if cfg["general"]["train_mode"]:
        train(cfg)
    else:
        test(cfg)


if __name__ == "__main__":
    main()
