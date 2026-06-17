import io
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pathlib

# graph_tool need to be imported before torch
try:
    import graph_tool.all as gt
except Exception as exc:
    print(f"Graph tool unavailable, continuing without it: {exc}")
import torch
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_fabric.plugins.io.torch_io import TorchCheckpointIO

import wandb
import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from diffusion_model_sparse import DiscreteDenoisingDiffusion
from metrics.molecular_metrics import TrainMolecularMetricsDiscrete
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from diffusion.extra_features_molecular import ExtraMolecularFeatures
from sparse_diffusion.metrics.sampling_metrics import SamplingMetrics

# debug for multi-gpu
import resource

resource.setrlimit(
    resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
)

import pickle


class _TeeIO(io.TextIOBase):
    """Mirror stdout/stderr to a file (subclass TextIOBase so Rich/PL don't treat .write as read-only)."""

    def __init__(self, *streams):
        super().__init__()
        self._streams = streams

    @property
    def encoding(self):
        return getattr(self._streams[0], "encoding", None) or "utf-8"

    def write(self, s):
        if not s:
            return 0
        for stream in self._streams:
            stream.write(s)
            stream.flush()
        return len(s)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def fileno(self):
        return self._streams[0].fileno()

    def isatty(self):
        return getattr(self._streams[0], "isatty", lambda: False)()


def _attach_repo_run_log(cfg: DictConfig) -> None:
    """Append stdout/stderr to repo root ``logs/{general.name}.log`` (disable with SPARSEDIFF_NO_RUN_LOG=1)."""
    if os.environ.get("SPARSEDIFF_NO_RUN_LOG", "").lower() in ("1", "true", "yes"):
        return
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    log_dir = repo_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe = str(cfg.general.name).replace("/", "_").replace("\\", "_")
    path = log_dir / f"{safe}.log"
    f = open(path, "a", encoding="utf-8")
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = _TeeIO(sys.__stdout__, f)
    sys.stderr = _TeeIO(sys.__stderr__, f)
    import atexit

    def _close_run_log():
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        f.close()

    atexit.register(_close_run_log)
    sys.__stdout__.write(f"[run log also at {path}]\n")
    sys.__stdout__.flush()




class TrustedTorchCheckpointIO(TorchCheckpointIO):
    """Checkpoint IO for local trusted Lightning checkpoints under PyTorch 2.6."""

    def load_checkpoint(self, path, map_location=lambda storage, loc: storage, weights_only=None):
        return super().load_checkpoint(path, map_location=map_location, weights_only=False)


def _load_ckpt_avoid_ddp(ckpt_path):
    """Load checkpoint without triggering DDP's init_process_group (for single-process test_only)."""
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        pass
    # Checkpoint 里含有 DDP 等对象时，临时把 DDP.__setstate__ 改为 no-op 再加载
    import torch.nn.parallel.distributed as dist_module
    _orig_setstate = getattr(dist_module.DistributedDataParallel, "__setstate__", None)

    def _noop_setstate(self, state):
        pass

    try:
        dist_module.DistributedDataParallel.__setstate__ = _noop_setstate
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    finally:
        if _orig_setstate is not None:
            dist_module.DistributedDataParallel.__setstate__ = _orig_setstate
    return ckpt


def _resolve_ckpt_path(ckpt_path):
    """Resolve checkpoint path after Hydra chdir (relative paths are from launch cwd)."""
    if not ckpt_path:
        return ckpt_path
    path = pathlib.Path(ckpt_path)
    if path.is_file():
        return str(path.resolve())
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    for base in (pathlib.Path(get_original_cwd()), repo_root):
        candidate = (base / path).resolve()
        if candidate.is_file():
            return str(candidate)
    return ckpt_path


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    _attach_repo_run_log(cfg)
    dataset_config = cfg["dataset"]
    pl.seed_everything(cfg.train.seed)

    print("creating datasets")
    if dataset_config["name"] in ["sbm", "comm20", "planar", "ego", "acm_subgraphs", "imdb_subgraphs", "pubmed_subgraphs", "pubmed_baseline_subgraphs", "pubmed_pgb_subgraphs", "dblp_subgraphs", "hghg_subgraphs", "dblp_single"]:
        from datasets.spectre_dataset_pyg import (
            SBMDataModule,
            Comm20DataModule,
            EgoDataModule,
            PlanarDataModule,
            SpectreDatasetInfos,
        )
        if dataset_config["name"] in ["acm_subgraphs", "imdb_subgraphs", "pubmed_subgraphs", "pubmed_baseline_subgraphs", "pubmed_pgb_subgraphs", "dblp_subgraphs"]:
            from datasets.acm_subgraphs_dataset import ACMSubgraphsDataModule, ACMSubgraphsInfos
            datamodule = ACMSubgraphsDataModule(cfg)
            dataset_infos = ACMSubgraphsInfos(datamodule)
            dataset_infos.datamodule = datamodule  # 供 cond_edge_gen_fixed_nodes 测试采样时取固定图（与验证一致）
            train_metrics = TrainAbstractMetricsDiscrete()
            domain_features = DummyExtraFeatures()
            dataloaders = datamodule.dataloaders
        elif dataset_config["name"] == "hghg_subgraphs":
            from datasets.hghg_subgraphs_dataset import HGHGSubgraphsDataModule, HGHGSubgraphsInfos
            datamodule = HGHGSubgraphsDataModule(cfg)
            dataset_infos = HGHGSubgraphsInfos(datamodule)
            dataset_infos.datamodule = datamodule
            train_metrics = TrainAbstractMetricsDiscrete()
            domain_features = DummyExtraFeatures()
            dataloaders = datamodule.dataloaders
        elif dataset_config["name"] == "dblp_single":
            from datasets.dblp_single_dataset import DBLPSingleDataModule, DBLPSingleInfos
            datamodule = DBLPSingleDataModule(cfg)
            dataset_infos = DBLPSingleInfos(datamodule)
            dataset_infos.datamodule = datamodule  # 供 cond_edge_gen_fixed_nodes 测试采样时取固定图
            train_metrics = TrainAbstractMetricsDiscrete()
            domain_features = DummyExtraFeatures()
            dataloaders = datamodule.dataloaders
        else:

            if dataset_config["name"] == "sbm":
                datamodule = SBMDataModule(cfg)
            elif dataset_config["name"] == "comm20":
                datamodule = Comm20DataModule(cfg)
            elif dataset_config["name"] == "ego":
                datamodule = EgoDataModule(cfg)
            else:
                datamodule = PlanarDataModule(cfg)

            dataset_infos = SpectreDatasetInfos(datamodule)
            train_metrics = TrainAbstractMetricsDiscrete()
            domain_features = DummyExtraFeatures()
            dataloaders = datamodule.dataloaders

    elif dataset_config["name"] == "protein":
        from datasets import protein_dataset

        datamodule = protein_dataset.ProteinDataModule(cfg)
        dataset_infos = protein_dataset.ProteinInfos(datamodule=datamodule)
        train_metrics = TrainAbstractMetricsDiscrete()
        domain_features = DummyExtraFeatures()
        dataloaders = datamodule.dataloaders

    elif dataset_config["name"] == "point_cloud":
        from datasets import point_cloud_dataset

        datamodule = point_cloud_dataset.PointCloudDataModule(cfg)
        dataset_infos = point_cloud_dataset.PointCloudInfos(datamodule=datamodule)
        train_metrics = TrainAbstractMetricsDiscrete()
        domain_features = DummyExtraFeatures()
        dataloaders = datamodule.dataloaders

    elif dataset_config["name"] in ["qm9", "guacamol", "moses"]:
        if dataset_config["name"] == "qm9":
            from datasets import qm9_dataset

            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9Infos(datamodule=datamodule, cfg=cfg)

        elif dataset_config["name"] == "guacamol":
            from datasets import guacamol_dataset

            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.GuacamolInfos(datamodule, cfg)

        elif dataset_config.name == "moses":
            from datasets import moses_dataset

            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MosesInfos(datamodule, cfg)
        else:
            raise ValueError("Dataset not implemented")

        dataloaders = None

        if cfg.model.extra_features is not None:
            # domain_features = DummyExtraFeatures()
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            domain_features = DummyExtraFeatures()

        train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    ef = cfg.model.extra_features
    edge_f = cfg.model.edge_features
    extra_features = (
        ExtraFeatures(
            eigenfeatures=cfg.model.eigenfeatures,
            edge_features_type=edge_f,
            dataset_info=dataset_infos,
            num_eigenvectors=cfg.model.num_eigenvectors,
            num_eigenvalues=cfg.model.num_eigenvalues,
            num_degree=cfg.model.num_degree,
            dist_feat=cfg.model.dist_feat,
            use_positional=cfg.model.positional_encoding,
        )
        if ef is not None
        else DummyExtraFeatures()
    )

    dataset_infos.compute_input_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features,
    )

    val_sampling_metrics = SamplingMetrics(
        dataset_infos, test=False, dataloaders=dataloaders
    )
    test_sampling_metrics = SamplingMetrics(
        dataset_infos, test=True, dataloaders=dataloaders
    )
    model_kwargs = {
        "dataset_infos": dataset_infos,
        "train_metrics": train_metrics,
        "extra_features": extra_features,
        "domain_features": domain_features,
        "val_sampling_metrics": val_sampling_metrics,
        "test_sampling_metrics": test_sampling_metrics,
    }

    utils.create_folders(cfg)

    print("creating model")
    model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)

    callbacks = []
    if cfg.train.save_model:
        checkpoint_base = os.path.join(
            utils.get_model_output_root(), "checkpoints", cfg.general.name
        )
        # 统一规则：默认按 val/epoch_NLL 选 best；不每 epoch 验证时可在 experiment 里设 checkpoint_monitor: train_epoch/NLL
        checkpoint_monitor = getattr(cfg.general, "checkpoint_monitor", "val/epoch_NLL")
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_base,
            filename="{epoch}",
            monitor=checkpoint_monitor,
            save_top_k=15,
            mode="min",
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

    name = cfg.general.name
    if name == "debug":
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    num_devices = int(cfg.general.gpus) if use_gpu else 1
    heterogeneous = getattr(dataset_infos, "heterogeneous", False)
    # 仅多卡时用 DDP。单卡用 auto（单进程、无 DDP），避免 DDP 下 edge_only 等路径出现 unused parameter 报错。
    if num_devices > 1:
        strategy = (
            "ddp_find_unused_parameters_true"
            if heterogeneous
            else "ddp"
        )
    else:
        strategy = "auto"
    
    enable_validation = getattr(cfg.general, "enable_validation", True)
    skip_first_val = getattr(cfg.general, "skip_first_validation", False)
    # wandb 在线时使用 WandbLogger，便于 checkpoint 与 loss 等指标上传
    # DDP 下每个进程都会跑 main：仅在 rank0 初始化 wandb，避免重复 run / 重复同步
    use_wandb = getattr(cfg.general, "wandb", "online") not in ("disabled", "off")
    _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if use_wandb and not cfg.general.test_only and _local_rank == 0:
        utils.setup_wandb(cfg)
        from pytorch_lightning.loggers import WandbLogger
        pl_logger = WandbLogger(project=f"sparse_{cfg.dataset.name}", name=cfg.general.name, experiment=wandb.run, log_model=False)
    else:
        pl_logger = []
    trainer_plugins = [TrustedTorchCheckpointIO()] if bool(getattr(cfg.general, "resume_full", False)) else None
    trainer = pl.Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy=strategy,
        accelerator="gpu" if use_gpu else "cpu",
        devices=cfg.general.gpus if use_gpu else 1,
        val_check_interval=cfg.general.val_check_interval,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=cfg.general.name == "debug",
        callbacks=callbacks,
        log_every_n_steps=getattr(cfg.general, "log_every_n_steps", 50) if name != "debug" else 1,
        enable_progress_bar=False,
        logger=pl_logger,
        limit_val_batches=0.0 if not enable_validation else 1.0,
        num_sanity_val_steps=0 if (not enable_validation or skip_first_val) else 2,
        plugins=trainer_plugins,
    )

    try:
        if not cfg.general.test_only and not cfg.general.generated_path:
            # 训练模式：若提供 general.resume，则以 strict=False 方式只加载模型权重进行 warm-start，
            # 避免因为新增/删除模块导致 Lightning 严格恢复 state_dict 时报 Unexpected key(s) 错误。
            resume_ckpt_path = None
            if cfg.general.resume:
                ckpt_path = _resolve_ckpt_path(cfg.general.resume)
                if os.path.isfile(ckpt_path):
                    if bool(getattr(cfg.general, "resume_full", False)):
                        resume_ckpt_path = ckpt_path
                        try:
                            from omegaconf import DictConfig, ListConfig
                            from omegaconf.base import ContainerMetadata, Metadata
                            from omegaconf.nodes import AnyNode, BooleanNode, BytesNode, EnumNode, FloatNode, IntegerNode, StringNode
                            torch.serialization.add_safe_globals([
                                DictConfig, ListConfig, ContainerMetadata, Metadata,
                                AnyNode, BooleanNode, BytesNode, EnumNode, FloatNode, IntegerNode, StringNode,
                            ])
                        except Exception as exc:
                            print(f"[WARN] Could not register OmegaConf safe globals for full resume: {exc}")
                        print(f"[INFO] Full resume from checkpoint: {ckpt_path}")
                    else:
                        print(f"[INFO] Warm-start from checkpoint (strict=False): {ckpt_path}")
                        try:
                            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                        except TypeError:
                            ckpt = torch.load(ckpt_path, map_location="cpu")
                        state_dict = ckpt.get("state_dict", ckpt)
                        missing, unexp = model.load_state_dict(state_dict, strict=False)
                        if missing or unexp:
                            print("[INFO] Loaded ckpt with strict=False; missing keys:", missing, "unexpected keys:", unexp)
                else:
                    print(f"[WARN] resume checkpoint not found: {ckpt_path}, training from scratch.")
            trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt_path)
            if cfg.general.name != "debug" and getattr(cfg.general, "run_test_after_train", True):
                trainer.test(model, datamodule=datamodule)
        else:
            # Start by evaluating test_only_path (respect train.seed for multi-seed test sweeps)
            pl.seed_everything(cfg.train.seed)
            # 用 strict=False 加载，以兼容缺少新模块（如 edge_fusion）的旧 ckpt，便于复用已训练权重
            ckpt_path = _resolve_ckpt_path(cfg.general.test_only)
            if ckpt_path != cfg.general.test_only:
                print(f"[INFO] Resolved checkpoint path: {ckpt_path}")
            if ckpt_path and os.path.isfile(ckpt_path):
                # 若 ckpt 含 DDP 等分布式对象，直接 torch.load 会触发 init_process_group；
                # 单进程 test 时先临时把 DDP.__setstate__ 换成 no-op 再加载
                ckpt = _load_ckpt_avoid_ddp(ckpt_path)
                if "state_dict" in ckpt:
                    missing, unexp = model.load_state_dict(ckpt["state_dict"], strict=False)
                    if missing or unexp:
                        print("Loaded ckpt with strict=False; missing:", missing, "unexpected:", unexp)
                trainer.test(model, datamodule=datamodule)
            else:
                trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)
            if cfg.general.evaluate_all_checkpoints:
                directory = pathlib.Path(ckpt_path).parents[0]
                print("Directory:", directory)
                files_list = os.listdir(directory)
                for file in files_list:
                    if ".ckpt" in file:
                        ckpt_path = os.path.join(directory, file)
                        if ckpt_path == cfg.general.test_only:
                            continue
                        print("Loading checkpoint", ckpt_path)
                        utils.setup_wandb(cfg)
                        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)
    except KeyboardInterrupt:
        print("Training interrupted by user (KeyboardInterrupt).")
        raise
    except Exception:
        # 异常退出不视为正常结束，不主动调用 wandb.finish()，让 wandb 将本次 run 标记为 crashed
        print("Training crashed with an exception; not calling wandb.finish() as success.")
        raise
    else:
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
