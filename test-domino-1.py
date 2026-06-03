#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_prepare_surface_data_wsl_v2.py

WSL 版 Ahmed Body 数据预处理脚本。

重要修改：
    不再依赖 physicsnemo.utils.domino.utils。
    直接在本脚本中实现：
        get_filenames
        get_node_to_elem
        get_fields

默认数据路径：
    /home/administrator/physicsnemo_data/physicsnemo_ahmed_body_dataset_vv1/dataset

用法：
    conda activate pnemo
    cd ~/physicsnemo_test

    python 01_prepare_surface_data_wsl_v2.py --workers 1

如果能跑通，再加并行：
    python 01_prepare_surface_data_wsl_v2.py --workers 4

重新生成：
    python 01_prepare_surface_data_wsl_v2.py --workers 4 --overwrite
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import vtk
from tqdm import tqdm
from vtk.util.numpy_support import vtk_to_numpy


# =========================
# 配置区
# =========================

DEFAULT_DATA_DIR = Path(
    "/home/administrator/physicsnemo_data/physicsnemo_ahmed_body_dataset_vv1/dataset"
)

VOLUME_VARS = ["p"]
SURFACE_VARS = ["p", "wallShearStress"]

GLOBAL_PARAMS_TYPES = {
    "inlet_velocity": "vector",
    "air_density": "scalar",
}

GLOBAL_PARAMS_REFERENCE = {
    "inlet_velocity": [50.0],
    "air_density": 1.226,
}


# =========================
# 替代 physicsnemo.utils.domino.utils 的三个函数
# =========================

def get_filenames(data_path: str | Path) -> list[str]:
    """返回目录下所有 .vtp 文件名。"""
    data_path = Path(data_path)
    files = sorted([p.name for p in data_path.iterdir() if p.is_file() and p.suffix.lower() == ".vtp"])
    if not files:
        raise FileNotFoundError(f"没有在目录中找到 .vtp 文件: {data_path}")
    return files


def get_node_to_elem(polydata: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """
    把点数据 PointData 转成单元数据 CellData。

    Ahmed body 的 VTP 里有些物理场可能存在 PointData 中，
    DoMINO 预处理希望得到每个表面单元中心对应的物理场，
    所以这里用 vtkPointDataToCellData 做平均映射。

    同时保留原有 CellData。
    """
    converter = vtk.vtkPointDataToCellData()
    converter.SetInputData(polydata)
    converter.PassPointDataOff()
    converter.Update()
    out = converter.GetOutput()

    # 如果原始 polydata 本身已经有 CellData，vtkPointDataToCellData 通常会保留；
    # 这里额外补充一遍，避免某些 VTK 版本行为不同。
    original_cell_data = polydata.GetCellData()
    output_cell_data = out.GetCellData()
    if original_cell_data is not None:
        for i in range(original_cell_data.GetNumberOfArrays()):
            arr = original_cell_data.GetArray(i)
            if arr is None:
                continue
            name = arr.GetName()
            if name and output_cell_data.GetArray(name) is None:
                output_cell_data.AddArray(arr)
    return out


def _find_vtk_array(data_obj: vtk.vtkDataSetAttributes, name: str):
    """按字段名查找 VTK 数组，找不到则返回 None。"""
    arr = data_obj.GetArray(name)
    if arr is not None:
        return arr

    # 兼容大小写或字段名轻微差异。
    lower_name = name.lower()
    for i in range(data_obj.GetNumberOfArrays()):
        candidate = data_obj.GetArray(i)
        if candidate is None or candidate.GetName() is None:
            continue
        if candidate.GetName().lower() == lower_name:
            return candidate
    return None


def get_fields(cell_data: vtk.vtkDataSetAttributes, variables: list[str]) -> list[np.ndarray]:
    """
    从 CellData 中提取变量，并转成二维 numpy 数组。

    标量 p: shape = [num_cells, 1]
    向量 wallShearStress: shape = [num_cells, 3]
    最后外部会 concatenate 成 [num_cells, 4]
    """
    arrays: list[np.ndarray] = []
    available = [
        cell_data.GetArray(i).GetName()
        for i in range(cell_data.GetNumberOfArrays())
        if cell_data.GetArray(i) is not None
    ]

    for var in variables:
        vtk_arr = _find_vtk_array(cell_data, var)
        if vtk_arr is None:
            raise KeyError(
                f"在 VTP CellData 中找不到字段: {var}\n"
                f"当前可用字段: {available}\n"
                f"如果字段在 PointData 中，本脚本会先尝试 PointDataToCellData；"
                f"如果仍找不到，说明该 VTP 文件字段名和预期不一致。"
            )
        arr = vtk_to_numpy(vtk_arr)
        if arr.ndim == 1:
            arr = arr[:, None]
        arrays.append(arr.astype(np.float32))
    return arrays


# =========================
# 通用工具
# =========================

def dataset_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024**3)


def setup_environment(data_dir: Path):
    print("=== Environment Setup ===")
    print(f"DATA_DIR = {data_dir}")

    dataset_paths = {split: data_dir / split for split in ["train", "validation", "test"]}
    info_paths = {split: data_dir / f"{split}_info" for split in dataset_paths}
    stl_paths = {split: data_dir / f"{split}_stl_files" for split in dataset_paths}
    surface_paths = {split: data_dir / f"{split}_prepared_surface_data" for split in dataset_paths}

    for split in dataset_paths:
        print(f"\n[{split}]")
        print(f"  vtp : {dataset_paths[split]}")
        print(f"  info: {info_paths[split]}")
        print(f"  stl : {stl_paths[split]}")
        print(f"  out : {surface_paths[split]}")

    return dataset_paths, info_paths, stl_paths, surface_paths


def read_velocity(info_path: Path) -> float:
    """从 *_info.txt 中读取 Velocity。"""
    with open(info_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if "Velocity" in line:
                return float(line.split(":")[1].strip())
    raise ValueError(f"没有在 info 文件中找到 Velocity: {info_path}")


def safe_unit_normals(normals: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(normals, axis=1)
    norm[norm == 0.0] = 1.0
    return normals / norm[:, None]


class OpenFoamAhmedBodySurfaceDataset:
    """读取一个 split 中的 VTP + STL + info，并转换成 DoMINO 需要的 numpy 字典。"""

    def __init__(
        self,
        data_path: str | Path,
        info_path: str | Path,
        stl_path: str | Path,
        surface_variables: list[str] | None = None,
        volume_variables: list[str] | None = None,
        global_params_types: dict[str, str] | None = None,
        global_params_reference: dict[str, Any] | None = None,
        shuffle_files: bool = False,
    ):
        self.data_path = Path(data_path).expanduser().resolve()
        self.stl_path = Path(stl_path).expanduser().resolve()
        self.info_path = Path(info_path).expanduser().resolve()

        if not self.data_path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.data_path}")
        if not self.stl_path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.stl_path}")
        if not self.info_path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.info_path}")

        self.filenames = get_filenames(self.data_path)
        if shuffle_files:
            random.shuffle(self.filenames)

        self.surface_variables = surface_variables or ["p", "wallShearStress"]
        self.volume_variables = volume_variables or ["p"]
        self.global_params_types = global_params_types or GLOBAL_PARAMS_TYPES
        self.global_params_reference = global_params_reference or GLOBAL_PARAMS_REFERENCE

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        cfd_filename = self.filenames[idx]
        car_path = self.data_path / cfd_filename
        case_stem = car_path.stem

        stl_path = self.stl_path / f"{case_stem}.stl"
        info_path = self.info_path / f"{case_stem}_info.txt"

        if not car_path.exists():
            raise FileNotFoundError(car_path)
        if not stl_path.exists():
            raise FileNotFoundError(stl_path)
        if not info_path.exists():
            raise FileNotFoundError(info_path)

        velocity = read_velocity(info_path)
        air_density = float(self.global_params_reference["air_density"])

        # 1) STL 几何
        mesh_stl = pv.read(str(stl_path))
        if mesh_stl.n_cells == 0:
            raise ValueError(f"STL 没有单元: {stl_path}")

        stl_faces = mesh_stl.faces.reshape(-1, 4)[:, 1:]
        stl_sizes = np.asarray(
            mesh_stl.compute_cell_sizes(length=False, area=True, volume=False).cell_data["Area"]
        )

        # 2) VTP 表面 CFD 结果
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(car_path))
        reader.Update()
        polydata = reader.GetOutput()
        if polydata is None or polydata.GetNumberOfCells() == 0:
            raise ValueError(f"VTP 没有单元: {car_path}")

        poly_cell = get_node_to_elem(polydata)
        cell_data = poly_cell.GetCellData()
        surface_fields = np.concatenate(get_fields(cell_data, self.surface_variables), axis=-1)

        # 按 notebook 中的方式无量纲化：p 和 wallShearStress 均除以 rho * U^2
        surface_fields = surface_fields / (air_density * velocity**2)

        mesh = pv.wrap(polydata)
        surface_sizes = np.asarray(
            mesh.compute_cell_sizes(length=False, area=True, volume=False).cell_data["Area"]
        )
        surface_normals = safe_unit_normals(np.asarray(mesh.cell_normals))

        # 3) 全局参数
        global_params_reference_list: list[float] = []
        for name, param_type in self.global_params_types.items():
            if param_type == "vector":
                global_params_reference_list.extend(self.global_params_reference[name])
            elif param_type == "scalar":
                global_params_reference_list.append(self.global_params_reference[name])
            else:
                raise ValueError(f"Unsupported global parameter type: {name}={param_type}")
        global_params_reference = np.array(global_params_reference_list, dtype=np.float32)

        global_params_values_list: list[float] = []
        for key in self.global_params_types.keys():
            if key == "inlet_velocity":
                global_params_values_list.append(velocity)
            elif key == "air_density":
                global_params_values_list.append(air_density)
            else:
                raise ValueError(f"Unsupported global parameter: {key}")
        global_params_values = np.array(global_params_values_list, dtype=np.float32)

        return {
            "stl_coordinates": np.asarray(mesh_stl.points, dtype=np.float32),
            "stl_centers": np.asarray(mesh_stl.cell_centers().points, dtype=np.float32),
            "stl_faces": stl_faces.flatten().astype(np.float32),
            "stl_areas": stl_sizes.astype(np.float32),
            "surface_mesh_centers": np.asarray(mesh.cell_centers().points, dtype=np.float32),
            "surface_normals": surface_normals.astype(np.float32),
            "surface_areas": surface_sizes.astype(np.float32),
            "volume_fields": None,
            "volume_mesh_centers": None,
            "surface_fields": surface_fields.astype(np.float32),
            "filename": cfd_filename,
            "global_params_values": global_params_values,
            "global_params_reference": global_params_reference,
        }


def process_file(args_tuple):
    fname, dataset_kwargs, output_path = args_tuple
    try:
        fm_data = OpenFoamAhmedBodySurfaceDataset(**dataset_kwargs)
        full_path = fm_data.data_path / fname
        output_file = Path(output_path) / f"{Path(fname).stem}.npy"

        if output_file.exists():
            return "skipped", fname, "already exists"
        if (not full_path.exists()) or full_path.stat().st_size == 0:
            return "skipped", fname, "missing/empty"

        idx = fm_data.filenames.index(fname)
        data = fm_data[idx]
        np.save(output_file, data)
        return "processed", fname, ""
    except Exception as e:
        return "failed", fname, repr(e)


def process_surface_data_batch(dataset_paths, info_paths, stl_paths, surface_paths, num_workers: int):
    for path in surface_paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)

    print("\n=== Starting Processing ===")
    for split, dataset_path in dataset_paths.items():
        surface_path = Path(surface_paths[split])
        surface_path.mkdir(parents=True, exist_ok=True)

        fm_data = OpenFoamAhmedBodySurfaceDataset(
            data_path=dataset_path,
            info_path=info_paths[split],
            stl_path=stl_paths[split],
            surface_variables=SURFACE_VARS,
            volume_variables=VOLUME_VARS,
            global_params_types=GLOBAL_PARAMS_TYPES,
            global_params_reference=GLOBAL_PARAMS_REFERENCE,
        )
        file_list = [fname for fname in fm_data.filenames if fname.endswith(".vtp")]
        print(f"\nProcessing {split}: {len(file_list)} files")
        print(f"  input : {dataset_path}")
        print(f"  output: {surface_path}")

        dataset_kwargs = dict(
            data_path=str(dataset_path),
            info_path=str(info_paths[split]),
            stl_path=str(stl_paths[split]),
            surface_variables=SURFACE_VARS,
            volume_variables=VOLUME_VARS,
            global_params_types=GLOBAL_PARAMS_TYPES,
            global_params_reference=GLOBAL_PARAMS_REFERENCE,
        )
        tasks = [(fname, dataset_kwargs, str(surface_path)) for fname in file_list]

        ok = skipped = failed = 0
        if num_workers <= 1:
            iterator = tqdm(tasks, total=len(tasks), desc=f"Processing {split}", dynamic_ncols=True)
            for task in iterator:
                status, fname, msg = process_file(task)
                ok += status == "processed"
                skipped += status == "skipped"
                failed += status == "failed"
                if status == "failed":
                    print(f"[FAILED] {fname}: {msg}")
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(process_file, task) for task in tasks]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split}", dynamic_ncols=True):
                    status, fname, msg = fut.result()
                    ok += status == "processed"
                    skipped += status == "skipped"
                    failed += status == "failed"
                    if status == "failed":
                        print(f"[FAILED] {fname}: {msg}")

        print(f"{split}: processed={ok}, skipped={skipped}, failed={failed}")
        if failed:
            raise RuntimeError(f"{split} split has {failed} failed files")

    print("\n=== All Processing Completed Successfully ===")


def summarize_outputs(surface_paths):
    print("\n=== Prepared .npy Summary ===")
    for split, path in surface_paths.items():
        files = sorted(Path(path).glob("*.npy"))
        print(f"{split:10s}: {len(files)} files at {path}")
        if not files:
            continue
        sample = np.load(files[0], allow_pickle=True).item()
        print(f"  sample: {files[0].name}")
        for key, val in sample.items():
            if hasattr(val, "shape"):
                print(f"    {key:24s} shape={str(val.shape):18s} dtype={getattr(val, 'dtype', '')}")
            else:
                print(f"    {key:24s} value={val}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Ahmed Body surface data for DoMINO, WSL self-contained version.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Ahmed Body dataset 根目录。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并行进程数。首次运行建议用 1，确认成功后再改 4。",
    )
    parser.add_argument("--overwrite", action="store_true", help="删除旧的 prepared_surface_data 后重新生成。")
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {data_dir}")

    print(f"Dataset size: {dataset_size_gb(data_dir):.2f} GB")
    dataset_paths, info_paths, stl_paths, surface_paths = setup_environment(data_dir)

    for split in dataset_paths:
        for p in [dataset_paths[split], info_paths[split], stl_paths[split]]:
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing required directory: {p}")

    if args.overwrite:
        for out_path in surface_paths.values():
            out_path = Path(out_path)
            if out_path.exists():
                print(f"Removing old output: {out_path}")
                shutil.rmtree(out_path)

    process_surface_data_batch(dataset_paths, info_paths, stl_paths, surface_paths, args.workers)
    summarize_outputs(surface_paths)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[错误] {e}", file=sys.stderr)
        sys.exit(1)