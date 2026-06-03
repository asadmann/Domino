# train_predict_domino_fixed_v2.py
# PhysicsNeMo DoMINO training + prediction script for Ahmed Body prepared dataset.
# Run inside the PhysicsNeMo Docker container.

import argparse
import csv
import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import pyvista as pv
import vtk
from vtk.util import numpy_support
from scipy.spatial import KDTree
from tqdm.auto import tqdm

from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import apex
except Exception:
    apex = None

from physicsnemo.distributed import DistributedManager
from physicsnemo.datapipes.cae.domino_datapipe import DoMINODataPipe
from physicsnemo.models.domino.model import DoMINO

# PhysicsNeMo 2.x moved signed_distance_field from physicsnemo.utils.sdf
# to physicsnemo.nn.functional.  Keep a small compatibility layer so this
# script also works with older package layouts when possible.
try:
    from physicsnemo.nn.functional import signed_distance_field as _pn_signed_distance_field
except Exception:  # pragma: no cover - compatibility fallback
    from physicsnemo.utils.sdf import signed_distance_field as _pn_signed_distance_field

# PhysicsNeMo 2.0.0 no longer exposes physicsnemo.utils.domino.utils.
# The small helpers used by this standalone script are reimplemented below.


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def dict_to_device(obj, device):
    """Recursively move tensors in a nested dict/list/tuple structure to device."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: dict_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [dict_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(dict_to_device(v, device) for v in obj)
    return obj


def create_directory(path):
    """Create a directory if it does not exist and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def calculate_center_of_mass(centers, weights):
    """Area-weighted center of mass for cell centers."""
    centers = np.asarray(centers, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    denom = np.sum(weights)
    if denom <= 0:
        return np.mean(centers, axis=0).astype(np.float32)
    return (np.sum(centers * weights[:, None], axis=0) / denom).astype(np.float32)


def create_grid(max_coord, min_coord, resolution):
    """Create a structured 3D grid with shape (nx, ny, nz, 3)."""
    max_coord = np.asarray(max_coord, dtype=np.float32)
    min_coord = np.asarray(min_coord, dtype=np.float32)
    nx, ny, nz = [int(v) for v in resolution]
    xs = np.linspace(min_coord[0], max_coord[0], nx, dtype=np.float32)
    ys = np.linspace(min_coord[1], max_coord[1], ny, dtype=np.float32)
    zs = np.linspace(min_coord[2], max_coord[2], nz, dtype=np.float32)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([xx, yy, zz], axis=-1).astype(np.float32)


def get_node_to_elem(polydata):
    """
    Convert point-data arrays on a VTK PolyData to cell-data arrays.

    The Ahmed Body VTP files may store CFD fields on points.  DoMINO inference
    here works on cell centers, so we convert point data to cell data.  Existing
    cell-data arrays are preserved by VTK in normal use.
    """
    converter = vtk.vtkPointDataToCellData()
    converter.SetInputData(polydata)
    converter.PassPointDataOn()
    converter.Update()
    return converter.GetOutput()


def get_fields(celldata, variable_names):
    """Fetch VTK cell-data arrays by name and return them as NumPy arrays."""
    fields = []
    available = [celldata.GetArrayName(i) for i in range(celldata.GetNumberOfArrays())]
    for name in variable_names:
        arr = celldata.GetArray(name)
        if arr is None:
            raise KeyError(f"VTK cell data field '{name}' not found. Available fields: {available}")
        np_arr = numpy_support.vtk_to_numpy(arr)
        if np_arr.ndim == 1:
            np_arr = np_arr[:, None]
        fields.append(np.asarray(np_arr, dtype=np.float32))
    return fields


def normalize(values, max_coord, min_coord, eps=1.0e-12):
    """Map coordinates from [min, max] to [-1, 1]."""
    values = np.asarray(values, dtype=np.float32)
    max_coord = np.asarray(max_coord, dtype=np.float32)
    min_coord = np.asarray(min_coord, dtype=np.float32)
    return (2.0 * (values - min_coord) / np.maximum(max_coord - min_coord, eps) - 1.0).astype(np.float32)


def unnormalize(values, max_value, min_value):
    """Inverse of the [-1, 1] min-max normalization used by the tutorial."""
    values = np.asarray(values, dtype=np.float32)
    max_value = np.asarray(max_value, dtype=np.float32)
    min_value = np.asarray(min_value, dtype=np.float32)
    return ((values + 1.0) * 0.5 * (max_value - min_value) + min_value).astype(np.float32)


def write_to_vtp(polydata, save_path):
    """Write a VTK PolyData object to a .vtp file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(save_path))
    writer.SetInputData(polydata)
    ok = writer.Write()
    if ok != 1:
        raise IOError(f"Failed to write VTP file: {save_path}")


def _unwrap_model(model):
    """Return the underlying model when wrapped by DistributedDataParallel."""
    return model.module if hasattr(model, "module") else model


def _object_to_state_dict(obj):
    """Convert common training objects to checkpoint-friendly state dicts."""
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    if isinstance(obj, dict):
        return {k: _object_to_state_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_object_to_state_dict(v) for v in obj]
    return obj


def save_checkpoint(save_dir, epoch=None, **kwargs):
    """
    Minimal replacement for the removed physicsnemo.launch.utils.save_checkpoint.

    The old tutorial code expects save_checkpoint(path, epoch=..., models=...,
    optimizer=..., scaler=...).  PhysicsNeMo 2.0.0 does not provide
    physicsnemo.launch, so we save a normal PyTorch .pt checkpoint here.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {}
    if epoch is not None:
        checkpoint["epoch"] = int(epoch)

    # Keep the key names that load_model_for_inference() understands.
    model_obj = kwargs.get("models", kwargs.get("model", None))
    if model_obj is not None:
        checkpoint["model_state_dict"] = _unwrap_model(model_obj).state_dict()

    optimizer_obj = kwargs.get("optimizer", None)
    if optimizer_obj is not None:
        checkpoint["optimizer_state_dict"] = optimizer_obj.state_dict()

    scaler_obj = kwargs.get("scaler", None)
    if scaler_obj is not None and hasattr(scaler_obj, "state_dict"):
        checkpoint["scaler_state_dict"] = scaler_obj.state_dict()

    # Preserve any extra values without breaking torch.save.
    for key, value in kwargs.items():
        if key in {"models", "model", "optimizer", "scaler"}:
            continue
        checkpoint[key] = _object_to_state_dict(value)

    filename = "checkpoint.pt" if epoch is None else f"checkpoint_epoch_{int(epoch) + 1:06d}.pt"
    save_path = save_dir / filename
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to: {save_path}")


def signed_distance_field(mesh_vertices, mesh_indices, query_points, **kwargs):
    """
    Compatibility wrapper around PhysicsNeMo 2.x signed_distance_field.

    Current PhysicsNeMo returns (sdf, closest_points).  The original DoMINO
    prediction snippet used a single return value and passed NumPy arrays.
    This wrapper accepts NumPy/Torch inputs, calls the current implementation,
    and returns a NumPy SDF array so the rest of this script can stay unchanged.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vertices = torch.as_tensor(mesh_vertices, dtype=torch.float32, device=device)
    indices = torch.as_tensor(mesh_indices, dtype=torch.int64, device=device)
    points = torch.as_tensor(query_points, dtype=torch.float32, device=device)

    with torch.no_grad():
        result = _pn_signed_distance_field(vertices, indices, points, **kwargs)

    sdf = result[0] if isinstance(result, (tuple, list)) else result
    return sdf.detach().cpu().numpy()


def build_config(args):
    data_dir = Path(args.data_dir).expanduser().resolve()
    project_name = args.project_name
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_dir = output_dir / "models"
    save_path = data_dir / args.pred_dir_name

    # These factors are from the NVIDIA Ahmed Body DoMINO tutorial notebook.
    # They are used for normalization/unnormalization of ["p", "wallShearStress"].
    surf_factors = np.array(
        [
            [0.41590533, 0.00170136, 0.00348770, 0.00238226],
            [-1.32783320, -0.00906987, -0.00507668, -0.00330993],
        ],
        dtype=np.float32,
    )

    model_type = "surface"
    surface_vars = ["p", "wallShearStress"]
    num_surf_vars = 4

    air_density = 1.205
    global_params_types = {"inlet_velocity": "vector", "air_density": "scalar"}
    global_params_reference = {"inlet_velocity": [50.0], "air_density": 1.226}

    normalization = "min_max_scaling"
    integral_loss_scaling = 0
    geometry_encoding_type = "both"

    grid_resolution = [128, 64, 48]
    surface_points_sample = args.surface_points_sample
    num_surface_neighbors = 7

    bounding_box = SimpleNamespace(max=[0.5, 0.6, 0.6], min=[-2.5, -0.5, -0.5])
    bounding_box_surf = SimpleNamespace(max=[0.01, 0.6, 0.4], min=[-1.5, -0.01, -0.01])

    geometry_rep = SimpleNamespace(
        geo_conv=SimpleNamespace(
            base_neurons=32,
            base_neurons_in=1,
            base_neurons_out=1,
            fourier_features=False,
            num_modes=5,
            volume_radii=[0.1, 0.5],
            surface_radii=[0.05],
            volume_neighbors_in_radius=[128, 128],
            surface_neighbors_in_radius=[128],
            surface_hops=1,
            volume_hops=1,
            activation="relu",
        ),
        geo_processor=SimpleNamespace(
            base_filters=8,
            activation="relu",
            cross_attention=False,
            self_attention=False,
            processor_type="conv",
            volume_sdf_scaling_factor=[1.0],
            surface_sdf_scaling_factor=[1.0],
        ),
        geo_processor_sdf=SimpleNamespace(base_filters=8),
    )

    geometry_local = SimpleNamespace(
        volume_neighbors_in_radius=[128, 128],
        surface_neighbors_in_radius=[128],
        volume_radii=[0.05, 0.1],
        surface_radii=[0.05],
        base_layer=512,
    )

    dataset_base_kwargs = {
        "grid_resolution": grid_resolution,
        "surface_variables": surface_vars,
        "normalize_coordinates": True,
        "sampling": True,
        "sample_in_bbox": True,
        "volume_points_sample": 8192,
        "surface_points_sample": surface_points_sample,
        "geom_points_sample": args.geom_points_sample,
        "positional_encoding": False,
        "surface_factors": surf_factors,
        "scaling_type": normalization,
        "model_type": model_type,
        "bounding_box_dims": bounding_box,
        "bounding_box_dims_surf": bounding_box_surf,
        "num_surface_neighbors": num_surface_neighbors,
        "gpu_preprocessing": False,
    }

    model_kwargs = {
        "input_features": 3,
        "output_features_vol": None,
        "output_features_surf": num_surf_vars,
        "model_parameters": SimpleNamespace(
            activation="relu",
            interp_res=grid_resolution,
            surface_neighbors=num_surface_neighbors,
            use_surface_normals=True,
            use_surface_area=True,
            encode_parameters=False,
            positional_encoding=False,
            num_neighbors_surface=7,
            num_neighbors_volume=7,
            combine_volume_surface=False,
            integral_loss_scaling_factor=integral_loss_scaling,
            normalization=normalization,
            use_sdf_in_basis_func=True,
            geometry_encoding_type=geometry_encoding_type,
            geometry_rep=geometry_rep,
            nn_basis_functions=SimpleNamespace(
                base_layer=512,
                fourier_features=True,
                num_modes=5,
                activation="relu",
            ),
            position_encoder=SimpleNamespace(
                base_neurons=512,
                activation="relu",
                fourier_features=False,
                num_modes=5,
            ),
            parameter_model=SimpleNamespace(
                base_layer=512,
                fourier_features=False,
                num_modes=5,
                activation="relu",
            ),
            geometry_local=geometry_local,
            local_point_conv=SimpleNamespace(activation="relu"),
            aggregation_model=SimpleNamespace(base_layer=512, activation="relu"),
            model_type=model_type,
        ),
    }

    return SimpleNamespace(
        project_name=project_name,
        data_dir=data_dir,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        model_save_dir=checkpoint_dir,
        save_path=save_path,
        data_paths={
            "train": str(data_dir / "train_prepared_surface_data"),
            "val": str(data_dir / "validation_prepared_surface_data"),
            "test": str(data_dir / "test"),
            "test_info": str(data_dir / "test_info"),
            "test_stl": str(data_dir / "test_stl_files"),
        },
        surf_factors=surf_factors,
        batch_size=args.batch_size,
        min_lr=args.min_lr,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_interval=args.checkpoint_interval,
        loss_history_path=output_dir / "loss_history.csv",
        predict_summary_path=save_path / "predict_summary.csv",
        surface_vars=surface_vars,
        air_density=air_density,
        global_params_types=global_params_types,
        global_params_reference=global_params_reference,
        grid_resolution=grid_resolution,
        bounding_box_surf=bounding_box_surf,
        num_surface_neighbors=num_surface_neighbors,
        dataset_base_kwargs=dataset_base_kwargs,
        model_kwargs=model_kwargs,
    )


def check_dataset_paths(cfg):
    required = [
        cfg.data_dir,
        Path(cfg.data_paths["train"]),
        Path(cfg.data_paths["val"]),
        Path(cfg.data_paths["test"]),
        Path(cfg.data_paths["test_info"]),
        Path(cfg.data_paths["test_stl"]),
    ]
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "以下路径在当前 Docker 容器内不存在，请检查 -v 挂载路径或 --data-dir：\n"
            + "\n".join(missing)
        )


def mse_loss_fn(output, target, padded_value=-10):
    target = target.to(output.device)
    mask = torch.abs(target - padded_value) > 1e-3
    return (torch.sum(((output - target) ** 2) * mask) / torch.clamp(torch.sum(mask), min=1)).mean()


def squeeze_extra_case_dim(batch):
    """
    Prepared DoMINO .npy files and DataLoader may together produce shapes like:
        [B, 1, 2, 3]
        [B, 1, N, 3]
        [B, 1, 128, 64, 48]
    Current PhysicsNeMo DoMINO expects:
        [B, 2, 3]
        [B, N, 3]
        [B, 128, 64, 48]
    So remove the extra dimension at dim=1 when it is a singleton.
    """
    fixed = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim >= 3 and value.shape[1] == 1:
            fixed[key] = value.squeeze(1)
        else:
            fixed[key] = value
    return fixed



class PreparedNpyDataset:
    """
    Minimal dataset for PhysicsNeMo DoMINO prepared .npy files.
    It reads case*.npy files saved as dict objects.
    """

    def __init__(self, data_path, phase="train"):
        self.data_path = Path(data_path)

        # If data_path is the dataset root, enter the corresponding prepared-data folder.
        if not list(self.data_path.glob("*.npy")):
            phase_to_folder = {
                "train": "train_prepared_surface_data",
                "val": "validation_prepared_surface_data",
                "validation": "validation_prepared_surface_data",
                "test": "test_prepared_surface_data",
            }
            sub = phase_to_folder.get(phase, phase)
            candidate = self.data_path / sub
            if candidate.exists():
                self.data_path = candidate

        def sort_key(path):
            nums = re.findall(r"\d+", path.stem)
            return int(nums[-1]) if nums else path.stem

        self.files = sorted(self.data_path.glob("*.npy"), key=sort_key)

        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npy files found in: {self.data_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()

        # The prepared .npy files store numpy arrays.
        # Many fields already contain a leading singleton dimension [1, ...].
        # DataLoader will add the real batch dimension, so remove that stored
        # singleton dimension here to avoid shapes like [B, 1, ...].
        converted = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                if value.shape[0:1] == (1,):
                    value = np.squeeze(value, axis=0)

                if value.dtype.kind in {"f", "i", "u", "b"}:
                    converted[key] = torch.from_numpy(value)
                else:
                    converted[key] = value
            else:
                converted[key] = value

        return converted


def create_dataset(cfg, phase):
    import inspect
    from physicsnemo.datapipes.cae.domino_datapipe import DoMINODataConfig

    kwargs = dict(cfg.dataset_base_kwargs)

    # In PhysicsNeMo 2.0.0, model_type is an argument of DoMINODataPipe,
    # not DoMINODataConfig.
    model_type = kwargs.pop("model_type", "surface")

    valid_keys = set(inspect.signature(DoMINODataConfig).parameters.keys())
    valid_keys.discard("data_path")

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    dropped_kwargs = sorted(set(kwargs.keys()) - set(filtered_kwargs.keys()))

    if dropped_kwargs:
        print(f"[DoMINODataPipe] Ignored unsupported dataset kwargs: {dropped_kwargs}")

    datapipe = DoMINODataPipe(
        cfg.data_paths[phase],
        model_type=model_type,
        phase=phase,
        **filtered_kwargs,
    )

    base_dataset = PreparedNpyDataset(cfg.data_paths[phase], phase=phase)
    datapipe.set_dataset(base_dataset)

    print(
        f"[DoMINODataPipe] phase={phase}, "
        f"data_path={base_dataset.data_path}, "
        f"samples={len(base_dataset)}"
    )

    return datapipe


def create_dataloaders(cfg, rank, world_size):
    train_dataset = create_dataset(cfg, "train")
    val_dataset = create_dataset(cfg, "val")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, val_loader, train_sampler, val_sampler


def create_model(cfg, device, rank=0, world_size=1):
    model = DoMINO(**cfg.model_kwargs).to(device)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
        )
    return model


def append_loss_history(cfg, epoch, train_loss, val_loss, lr, is_best):
    """Append one epoch of training history to a CSV file."""
    path = Path(cfg.loss_history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "train_loss", "val_loss", "lr", "is_best"])
        writer.writerow([
            int(epoch) + 1,
            f"{float(train_loss):.10e}",
            f"{float(val_loss):.10e}",
            f"{float(lr):.10e}",
            int(bool(is_best)),
        ])


def save_predict_summary(cfg, outputs):
    """Save one-row-per-case force summary for prediction runs."""
    if not outputs:
        return

    path = Path(cfg.predict_summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for out in outputs:
        force_true = float(out.get("force_x_true", 0.0))
        force_pred = float(out.get("force_x_pred", 0.0))
        abs_error = abs(force_pred - force_true)
        rel_error = abs_error / (abs(force_true) + 1.0e-12)
        rows.append({
            "case": Path(out["save_path"]).stem.replace("_predicted", ""),
            "save_path": out["save_path"],
            "pres_x_pred": out.get("pres_x_pred"),
            "pres_x_true": out.get("pres_x_true"),
            "shear_x_pred": out.get("shear_x_pred"),
            "shear_x_true": out.get("shear_x_true"),
            "force_x_pred": force_pred,
            "force_x_true": force_true,
            "force_x_abs_error": abs_error,
            "force_x_rel_error": rel_error,
        })

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Prediction summary saved to: {path}")


def run_epoch(cfg, train_loader, val_loader, model, optimizer, scaler, device, epoch, best_vloss, rank, world_size):
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}") if rank == 0 else train_loader

    for batch in pbar:
        batch = squeeze_extra_case_dim(dict_to_device(batch, device))
        optimizer.zero_grad()

        with autocast():
            _, pred_surf = model(batch)
            loss = mse_loss_fn(pred_surf, batch["surface_fields"])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += float(loss.detach().item())
        if rank == 0:
            avg = train_loss / max(1, pbar.n + 1)
            pbar.set_postfix({"train_loss": f"{avg:.5e}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

    avg_train_loss = train_loss / max(1, len(train_loader))

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = squeeze_extra_case_dim(dict_to_device(batch, device))
            with autocast():
                _, pred_surf = model(batch)
                loss = mse_loss_fn(pred_surf, batch["surface_fields"])
            val_loss += float(loss.item())
    val_loss = val_loss / max(1, len(val_loader))

    if world_size > 1:
        vals = torch.tensor([avg_train_loss, val_loss], device=device)
        torch.distributed.all_reduce(vals, op=torch.distributed.ReduceOp.SUM)
        vals /= world_size
        avg_train_loss, val_loss = vals.tolist()

    is_best = val_loss < best_vloss

    if rank == 0:
        cp_args = {"models": model, "optimizer": optimizer, "scaler": scaler}
        if is_best:
            save_checkpoint(str(cfg.model_save_dir / "best_model"), **cp_args)
            print(f"Saved best checkpoint: val_loss={val_loss:.6e}")
        if (epoch + 1) % cfg.checkpoint_interval == 0:
            save_checkpoint(str(cfg.model_save_dir), epoch=epoch, **cp_args)

        lr = optimizer.param_groups[0]["lr"]
        append_loss_history(cfg, epoch, avg_train_loss, val_loss, lr, is_best)
        print(
            f"Epoch {epoch + 1}: train_loss={avg_train_loss:.6e}, "
            f"val_loss={val_loss:.6e}, is_best={int(is_best)}"
        )
        print(f"Loss history saved to: {cfg.loss_history_path}")

    return val_loss


def train_domino(cfg):
    check_dataset_paths(cfg)
    cfg.model_save_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12356")
    os.environ.setdefault("LOCAL_RANK", "0")

    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    dist = DistributedManager()
    device, rank, world_size = dist.device, dist.rank, dist.world_size

    if rank == 0:
        print(f"device={device}, rank={rank}, world_size={world_size}")
        print(f"data_dir={cfg.data_dir}")
        print(f"output_dir={cfg.output_dir}")

    train_loader, val_loader, train_sampler, _ = create_dataloaders(cfg, rank, world_size)

    if rank == 0:
        sample = squeeze_extra_case_dim(next(iter(train_loader)))
        print("\nSample batch keys:")
        for key, value in sample.items():
            print(f"  {key:30s} {type(value).__name__:12s} shape={getattr(value, 'shape', None)}")

    model = create_model(cfg, device, rank, world_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.min_lr,
    )
    if rank == 0:
        print("Using optimizer: torch.optim.Adam")
        print(f"Using scheduler: CosineAnnealingLR(T_max={cfg.epochs}, eta_min={cfg.min_lr})")

    scaler = GradScaler()
    best_vloss = float("inf")

    for epoch in range(cfg.epochs):
        if world_size > 1 and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        curr_vloss = run_epoch(
            cfg, train_loader, val_loader, model, optimizer, scaler,
            device, epoch, best_vloss, rank, world_size
        )
        scheduler.step()
        if rank == 0:
            print(f"Next epoch lr={scheduler.get_last_lr()[0]:.6e}")
        best_vloss = min(best_vloss, curr_vloss)

    return model


def find_checkpoint(cfg, user_ckpt=None):
    if user_ckpt:
        p = Path(user_ckpt).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"checkpoint 不存在：{p}")
        return p

    # Prefer the validation-best checkpoint.  This avoids accidentally using a
    # later periodic checkpoint whose validation loss is worse.
    best_ckpt = cfg.model_save_dir / "best_model" / "checkpoint.pt"
    if best_ckpt.exists():
        return best_ckpt

    candidates = []
    if cfg.model_save_dir.exists():
        candidates.extend(cfg.model_save_dir.rglob("*.pt"))

    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"在 {cfg.model_save_dir} 下没有找到 .pt checkpoint。请先训练，或用 --checkpoint 指定。")
    return candidates[0]


def load_model_for_inference(cfg, device, checkpoint_path=None):
    model = create_model(cfg, device, 0, 1)
    ckpt = find_checkpoint(cfg, checkpoint_path)
    print(f"Loading checkpoint: {ckpt}")

    state = torch.load(str(ckpt), map_location=device)
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            state = state["model"]

    # If saved from DDP, remove "module." prefix.
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            new_state[k.replace("module.", "", 1)] = v
        state = new_state

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    return model


def read_velocity_from_info(info_path):
    with open(info_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Velocity" in line:
                return float(line.split(":", 1)[1].strip())
    raise ValueError(f"未在 info 文件中找到 Velocity：{info_path}")


def test_step(cfg, model, data_dict, device):
    """
    Inference step compatible with PhysicsNeMo 2.0.0 DoMINO.

    Older tutorial code manually called internal methods such as:
        model.geo_encoding_local(...)
        model.position_encoder(...)
        model.calculate_solution_with_neighbors(...)

    In the current DoMINO implementation, those internal attributes may not exist.
    Use the public forward interface instead:
        _, pred_surf = model(data_dict)
    """
    with torch.no_grad():
        data_dict = squeeze_extra_case_dim(dict_to_device(data_dict, device))

        global_params_values = data_dict["global_params_values"]
        global_params_reference = data_dict["global_params_reference"]

        def _extract_param(tensor, index):
            # Supports common shapes:
            # [B, 2, 1], [B, 1, 2], [B, 2]
            if tensor.ndim == 3:
                if tensor.shape[1] > index:
                    return tensor[:, index, :].reshape(tensor.shape[0], -1)[:, 0]
                if tensor.shape[2] > index:
                    return tensor[:, :, index].reshape(tensor.shape[0], -1)[:, 0]
            if tensor.ndim == 2:
                return tensor[:, index]
            if tensor.ndim == 1:
                return tensor[index].reshape(1)
            raise ValueError(f"Unsupported global parameter shape: {tensor.shape}")

        stream_velocity = _extract_param(global_params_values, 0)
        air_density = _extract_param(global_params_values, 1)

        _, pred_surf = model(data_dict)

        pred = pred_surf.detach().cpu().numpy()

        velocity_value = float(stream_velocity.detach().cpu().reshape(-1)[0])
        density_value = float(air_density.detach().cpu().reshape(-1)[0])

        pred = (
            unnormalize(pred, cfg.surf_factors[0], cfg.surf_factors[1])
            * velocity_value ** 2.0
            * density_value
        )

    return pred


def predict_one_case(cfg, model, device, vtp_path):
    vtp_path = Path(vtp_path)
    stem = vtp_path.stem
    tag_match = re.findall(r"(\w+?)(\d+)", stem)
    tag = int(tag_match[0][1]) if tag_match else stem

    stl_path = Path(cfg.data_paths["test_stl"]) / f"{stem}.stl"
    info_path = Path(cfg.data_paths["test_info"]) / f"{stem}_info.txt"

    if not stl_path.exists():
        raise FileNotFoundError(f"找不到对应 STL：{stl_path}")
    if not info_path.exists():
        raise FileNotFoundError(f"找不到对应 info：{info_path}")

    stream_velocity = read_velocity_from_info(info_path)

    mesh_stl = pv.read(str(stl_path))
    stl_vertices = mesh_stl.points
    stl_faces = np.array(mesh_stl.faces).reshape((-1, 4))[:, 1:]
    mesh_indices_flattened = stl_faces.flatten()
    length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))

    stl_sizes = np.array(mesh_stl.compute_cell_sizes(length=False, area=True, volume=False).cell_data["Area"], dtype=np.float32)
    stl_centers = np.array(mesh_stl.cell_centers().points, dtype=np.float32)
    center_of_mass = calculate_center_of_mass(stl_centers, stl_sizes)

    s_max = np.float32(np.asarray(cfg.bounding_box_surf.max))
    s_min = np.float32(np.asarray(cfg.bounding_box_surf.min))
    nx, ny, nz = cfg.grid_resolution

    surf_grid = create_grid(s_max, s_min, [nx, ny, nz])
    surf_grid_reshaped = surf_grid.reshape(nx * ny * nz, 3)
    sdf_surf_grid = signed_distance_field(
        stl_vertices,
        mesh_indices_flattened,
        surf_grid_reshaped,
        use_sign_winding_number=True,
    ).reshape(nx, ny, nz)

    surf_grid = np.float32(surf_grid)
    sdf_surf_grid = np.float32(sdf_surf_grid)
    surf_grid_max_min = np.float32(np.asarray([s_min, s_max]))

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    polydata_surf = reader.GetOutput()
    celldata_all = get_node_to_elem(polydata_surf)
    celldata = celldata_all.GetCellData()

    surface_fields = get_fields(celldata, cfg.surface_vars)
    surface_fields = np.concatenate(surface_fields, axis=-1)

    mesh = pv.PolyData(polydata_surf)
    surface_coordinates = np.array(mesh.cell_centers().points, dtype=np.float32)

    interp_func = KDTree(surface_coordinates)
    _, ii = interp_func.query(surface_coordinates, k=cfg.num_surface_neighbors)
    surface_neighbors = surface_coordinates[ii][:, 1:]

    surface_normals = np.array(mesh.cell_normals, dtype=np.float32)
    surface_normals = surface_normals / np.linalg.norm(surface_normals, axis=1)[:, None]

    surface_sizes = np.array(
        mesh.compute_cell_sizes(length=False, area=True, volume=False).cell_data["Area"],
        dtype=np.float32,
    )

    surface_neighbors_normals = surface_normals[ii][:, 1:]
    surface_neighbors_sizes = surface_sizes[ii][:, 1:]

    pos_surface_center_of_mass = surface_coordinates - center_of_mass
    surface_coordinates_norm = normalize(surface_coordinates, s_max, s_min)
    surface_neighbors_norm = normalize(surface_neighbors, s_max, s_min)
    surf_grid_norm = normalize(surf_grid, s_max, s_min)

    global_params_reference_list = []
    for name, typ in cfg.global_params_types.items():
        ref = cfg.global_params_reference[name]
        if typ == "vector":
            global_params_reference_list.extend(ref)
        elif typ == "scalar":
            global_params_reference_list.append(ref)
        else:
            raise ValueError(f"Unsupported global parameter type: {name}={typ}")
    global_params_reference = np.array(global_params_reference_list, dtype=np.float32)

    global_params_values = []
    for key in cfg.global_params_types.keys():
        if key == "inlet_velocity":
            global_params_values.append(stream_velocity)
        elif key == "air_density":
            global_params_values.append(cfg.air_density)
        else:
            raise ValueError(f"Unsupported global parameter: {key}")
    global_params_values = np.array(global_params_values, dtype=np.float32)

    data_dict = {
        "pos_surface_center_of_mass": np.float32(pos_surface_center_of_mass),
        "geometry_coordinates": np.float32(stl_vertices),
        "surf_grid": np.float32(surf_grid_norm),
        "sdf_surf_grid": np.float32(sdf_surf_grid),
        "surface_mesh_centers": np.float32(surface_coordinates_norm),
        "surface_mesh_neighbors": np.float32(surface_neighbors_norm),
        "surface_normals": np.float32(surface_normals),
        "surface_neighbors_normals": np.float32(surface_neighbors_normals),
        "surface_areas": np.float32(surface_sizes),
        "surface_neighbors_areas": np.float32(surface_neighbors_sizes),
        "surface_fields": np.float32(surface_fields),
        "surface_min_max": np.float32(surf_grid_max_min),
        "length_scale": np.array(length_scale, dtype=np.float32),
        "global_params_values": np.expand_dims(global_params_values, -1),
        "global_params_reference": np.expand_dims(global_params_reference, -1),
    }
    data_dict = {k: torch.from_numpy(np.expand_dims(np.float32(v), 0)) for k, v in data_dict.items()}

    prediction_surf = test_step(cfg, model, data_dict, device)

    cfg.save_path.mkdir(parents=True, exist_ok=True)
    vtp_pred_save_path = cfg.save_path / f"{stem}_predicted.vtp"

    surface_sizes_col = np.expand_dims(surface_sizes, -1)
    ps = prediction_surf
    sn = surface_normals
    sf = surface_fields
    ss = surface_sizes_col

    outputs = {
        "save_path": str(vtp_pred_save_path),
        "pres_x_pred": float(np.sum(ps[0, :, 0] * sn[:, 0] * ss[:, 0])),
        "shear_x_pred": float(np.sum(ps[0, :, 1] * ss[:, 0])),
        "pres_x_true": float(np.sum(sf[:, 0] * sn[:, 0] * ss[:, 0])),
        "shear_x_true": float(np.sum(sf[:, 1] * ss[:, 0])),
        "force_x_pred": float(np.sum(ps[0, :, 0] * sn[:, 0] * ss[:, 0] - ps[0, :, 1] * ss[:, 0])),
        "force_x_true": float(np.sum(sf[:, 0] * sn[:, 0] * ss[:, 0] - sf[:, 1] * ss[:, 0])),
    }

    arr = numpy_support.numpy_to_vtk(ps[0, :, 0:1])
    arr.SetName(f"{cfg.surface_vars[0]}Pred")
    celldata_all.GetCellData().AddArray(arr)

    arr = numpy_support.numpy_to_vtk(ps[0, :, 1:])
    arr.SetName(f"{cfg.surface_vars[1]}Pred")
    celldata_all.GetCellData().AddArray(arr)

    write_to_vtp(celldata_all, str(vtp_pred_save_path))
    return outputs


def predict_domino(cfg, checkpoint_path=None, max_cases=None):
    check_dataset_paths(cfg)

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12356")
    os.environ.setdefault("LOCAL_RANK", "0")

    if not DistributedManager.is_initialized():
        DistributedManager.initialize()

    dist = DistributedManager()
    device = dist.device

    model = load_model_for_inference(cfg, device, checkpoint_path)

    test_files = sorted(Path(cfg.data_paths["test"]).glob("*.vtp"))
    if max_cases is not None:
        test_files = test_files[:max_cases]

    if not test_files:
        raise FileNotFoundError(f"在 {cfg.data_paths['test']} 下没有找到 .vtp 测试文件")

    print(f"Found {len(test_files)} test VTP files.")
    all_outputs = []
    for vtp in test_files:
        print(f"\nPredicting: {vtp.name}")
        out = predict_one_case(cfg, model, device, vtp)
        all_outputs.append(out)

        force_abs_error = abs(out["force_x_pred"] - out["force_x_true"])
        force_rel_error = force_abs_error / (abs(out["force_x_true"]) + 1.0e-12)
        print(f"Saved: {out['save_path']}")
        print(
            f"force_x_pred={out['force_x_pred']:.6e}, "
            f"force_x_true={out['force_x_true']:.6e}, "
            f"abs_error={force_abs_error:.6e}, "
            f"rel_error={force_rel_error:.6%}"
        )

    save_predict_summary(cfg, all_outputs)
    return all_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict", "all"], default="all")
    parser.add_argument(
        "--data-dir",
        default="/home/administrator/physicsnemo_data/physicsnemo_ahmed_body_dataset_vv1/dataset",
        help="Dataset root inside the Docker container.",
    )
    parser.add_argument("--project-name", default="ahmed_body_dataset")
    parser.add_argument("--output-dir", default="./outputs/ahmed_body_dataset/4")
    parser.add_argument("--pred-dir-name", default="mesh_predictions_surf_final1")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--min-lr", type=float, default=1.0e-5)
    parser.add_argument("--checkpoint-interval", type=int, default=1)

    parser.add_argument("--surface-points-sample", type=int, default=8192)
    parser.add_argument("--geom-points-sample", type=int, default=40000)

    parser.add_argument("--checkpoint", default=None, help="Checkpoint path for prediction. If omitted, use latest .pt under output models.")
    parser.add_argument("--max-predict-cases", type=int, default=None)

    args = parser.parse_args()
    cfg = build_config(args)

    if args.mode in ["train", "all"]:
        train_domino(cfg)

    if args.mode in ["predict", "all"]:
        predict_domino(cfg, checkpoint_path=args.checkpoint, max_cases=args.max_predict_cases)


if __name__ == "__main__":
    main()
