"""nuScenes data pipeline for Safe-VLA training.

Loads nuScenes v1.0-mini JSON tables directly (no devkit dependency),
builds lookup indices for fast token-based access, and produces training
samples for a Vision-Language-Action model (Qwen2.5-VL).

Each sample contains:
  - A front-camera image (PIL, resized)
  - A DriveLM-style text prompt with ego speed & obstacle context
  - A target text with chain-of-thought rationale + waypoints JSON
  - Ground-truth future waypoints (x, y, v) in ego-local BEV frame
  - Nearby obstacle metadata in ego-local frame
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration helper
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Absolute or relative path to the YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as fh:
        config: Dict[str, Any] = yaml.safe_load(fh)
    logger.info("Loaded config from %s", config_path)
    return config


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def _quat_to_rotation_matrix(
    qw: float, qx: float, qy: float, qz: float
) -> np.ndarray:
    """Convert a unit quaternion to a 3x3 rotation matrix.

    Uses the Hamilton convention (scalar-first): q = (qw, qx, qy, qz).

    Args:
        qw: Scalar (real) component.
        qx: x component of the vector part.
        qy: y component of the vector part.
        qz: z component of the vector part.

    Returns:
        3x3 rotation matrix as a numpy float64 array.
    """
    norm = np.sqrt(qw**2 + qx**2 + qy **2 + qz**2)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    R = np.array([
         [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
         [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
         [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
       ])
    return R

def _global_to_ego(
    points_global: np.ndarray,
    ego_translation: np.ndarray,
    ego_rotation_matrix: np.ndarray,
) -> np.ndarray:
    """Transform 3-D points from the global frame to the ego-local frame.

    The ego frame has the ego vehicle at the origin with its heading along +x.

    Args:
        points_global: (N, 3) array of points in global coordinates.
        ego_translation: (3,) ego position in the global frame.
        ego_rotation_matrix: (3, 3) rotation from ego-local → global.

    Returns:
        (N, 3) array of points in the ego-local frame.
    """
    # So ideally we could have done:
    # p_global = r_ego * p_ego + t_ego
    # p_global - t_ego = r_ego * p*ego
    # p_ego = (r_ego)^-1 (p_global - t_ego)
    # but since a pure rotation matrix is an orthogonal matrix, it's inverse is simply the tranpose
    # p_ego = (r_ego_transpose) (p_global - t_ego)
    # but now since the nummpy arrays are of the shape (n,3) we can directly multiply (3,3) rotation matrix with (n,3) p_centered
    # so we'll have to tranpose that too, do the multiplication and tranpose it back
    # p_ego = ((r_ego_transpose) (p_centered_tranpose))_tranpose
    # p_ego =  (p_centered_tranpose)_tranpose *(r_ego_tranpose)_tranpose
    # p_ego = p_centered * r_ego   
    # and that's how we got the formula below
    points_centered = (points_global - ego_translation) @ ego_rotation_matrix

    return points_centered


def _yaw_from_quaternion(qw: float, qx: float, qy: float, qz: float) -> float:
    """Extract yaw (rotation about z-axis) from a quaternion.

    Args:
        qw: Scalar component.
        qx: x vector component.
        qy: y vector component.
        qz: z vector component.

    Returns:
        Yaw angle in radians.
    """
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy**2 + qz**2)
    yaw = np.atan2(siny_cosp, cosy_cosp)
    return yaw


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------

class NuScenesVLADataset(Dataset):
    """PyTorch dataset that yields Safe-VLA training samples from nuScenes.

    Each item is a dictionary containing an image, a DriveLM-style prompt,
    a target text with CoT rationale and waypoints, ground-truth future
    waypoints in ego-local BEV, nearby obstacles, and ego speed.

    Args:
        config: Parsed configuration dictionary (see ``load_config``).
        split: One of ``'train'`` or ``'val'``.
    """

    # Prompt template following the DriveLM style.
    _PROMPT_TEMPLATE: str = (
        "You are an autonomous driving AI. Analyze the front camera image "
        "and the current driving context.\n\n"
        "Current ego speed: {speed:.1f} m/s\n"
        "Nearby obstacles: {obstacle_summary}\n\n"
        "Task: Provide a brief chain-of-thought driving rationale, then "
        "predict the ego vehicle's future trajectory as 6 waypoints over the "
        "next 3 seconds (2 Hz) in ego-local BEV coordinates (x=forward, "
        "y=left).\n\n"
        "Output format:\n"
        "<rationale>Your reasoning here</rationale>\n"
        "<waypoints>[[x1,y1,v1], [x2,y2,v2], ..., [x6,y6,v6]]</waypoints>"
    )

    def __init__(self, config: Dict[str, Any], split: str = "train") -> None:
        super().__init__()
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        self._split = split
        data_cfg = config["data"]
        self._nuscenes_root = Path(data_cfg["nuscenes_root"])
        self._version = data_cfg["version"]
        self._camera = data_cfg["camera"]
        self._image_size: Tuple[int, int] = tuple(data_cfg["image_size"])  # (H, W)
        self._num_future_wps: int = data_cfg["num_future_waypoints"]
        self._wp_freq_hz: float = data_cfg["waypoint_freq_hz"]
        self._obstacle_radius: float = data_cfg["obstacle_radius_m"]
        self._train_ratio: float = data_cfg["train_split_ratio"]
        self._seed: int = data_cfg["seed"]

        # ---- Load JSON tables ------------------------------------------------
        table_dir = self._nuscenes_root / self._version
        self._scenes: List[Dict] = self._load_table(table_dir / "scene.json")
        self._samples: List[Dict] = self._load_table(table_dir / "sample.json")
        self._sample_data: List[Dict] = self._load_table(table_dir / "sample_data.json")
        self._ego_poses: List[Dict] = self._load_table(table_dir / "ego_pose.json")
        self._annotations: List[Dict] = self._load_table(table_dir / "sample_annotation.json")
        self._instances: List[Dict] = self._load_table(table_dir / "instance.json")
        self._categories: List[Dict] = self._load_table(table_dir / "category.json")

        # ---- Build fast token→record lookup dicts ----------------------------
        self._scene_by_token: Dict[str, Dict] = {r["token"]: r for r in self._scenes}
        self._sample_by_token: Dict[str, Dict] = {r["token"]: r for r in self._samples}
        self._sample_data_by_token: Dict[str, Dict] = {r["token"]: r for r in self._sample_data}
        self._ego_pose_by_token: Dict[str, Dict] = {r["token"]: r for r in self._ego_poses}
        self._instance_by_token: Dict[str, Dict] = {r["token"]: r for r in self._instances}
        self._category_by_token: Dict[str, Dict] = {r["token"]: r for r in self._categories}

        # sample_token → list of annotations
        self._anns_by_sample: Dict[str, List[Dict]] = {}
        for ann in self._annotations:
            self._anns_by_sample.setdefault(ann["sample_token"], []).append(ann)

        # sample_token → CAM_FRONT keyframe sample_data record
        self._cam_sd_by_sample: Dict[str, Dict] = {}
        for sd in self._sample_data:
            if sd.get("is_key_frame") and f"{self._camera}/" in sd.get("filename", ""):
                self._cam_sd_by_sample[sd["sample_token"]] = sd

        # ---- Train / val scene split -----------------------------------------
        rng = np.random.RandomState(self._seed)
        scene_tokens = [s["token"] for s in self._scenes]
        rng.shuffle(scene_tokens)
        n_train = max(1, int(len(scene_tokens) * self._train_ratio))
        if split == "train":
            split_scene_tokens = set(scene_tokens[:n_train])
        else:
            split_scene_tokens = set(scene_tokens[n_train:])

        logger.info(
            "Split '%s': %d / %d scenes", split, len(split_scene_tokens), len(scene_tokens)
        )

        # ---- Enumerate valid samples -----------------------------------------
        # A sample is valid if:
        #   1. It belongs to a scene in our split.
        #   2. It has a CAM_FRONT keyframe image.
        #   3. It has at least num_future_waypoints successive samples.
        self._valid_items: List[Tuple[str, str]] = []  # (sample_token, scene_token)
        for scene_token in split_scene_tokens:
            scene = self._scene_by_token[scene_token]
            # Walk the sample linked-list and collect tokens in order
            ordered_tokens: List[str] = []
            cur_token: str = scene["first_sample_token"]
            while cur_token:
                ordered_tokens.append(cur_token)
                cur_token = self._sample_by_token[cur_token].get("next", "")

            # Keep samples that have enough future successors
            for i, stok in enumerate(ordered_tokens):
                remaining = len(ordered_tokens) - i - 1
                if remaining < self._num_future_wps:
                    continue
                if stok not in self._cam_sd_by_sample:
                    logger.debug("Sample %s missing CAM_FRONT data; skipping.", stok)
                    continue
                self._valid_items.append((stok, scene_token))

        logger.info(
            "NuScenesVLADataset [%s]: %d valid samples from %d scenes",
            split,
            len(self._valid_items),
            len(split_scene_tokens),
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._valid_items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Fetch a single training sample.

        Args:
            idx: Index into the dataset.

        Returns:
            Dictionary with keys: ``image``, ``prompt``, ``target_text``,
            ``waypoints_gt``, ``obstacles``, ``ego_speed``, ``sample_token``.

        Raises:
            IndexError: If *idx* is out of range.
        """
        if idx < 0 or idx >= len(self._valid_items):
            raise IndexError(f"Index {idx} out of range [0, {len(self._valid_items)})")

        sample_token, scene_token = self._valid_items[idx]
        sample_rec = self._sample_by_token[sample_token]
        scene_rec = self._scene_by_token[scene_token]

        # ---- Image -----------------------------------------------------------
        image = self._load_image(sample_token)

        # ---- Current ego pose ------------------------------------------------
        ego_pose_cur = self._get_ego_pose_for_sample(sample_token)
        ego_trans_cur = np.array(ego_pose_cur["translation"], dtype=np.float64)
        ego_rot_cur = ego_pose_cur["rotation"]  # [qw, qx, qy, qz]
        ego_R_cur = _quat_to_rotation_matrix(*ego_rot_cur)

        # ---- Future waypoints in ego-local frame -----------------------------
        waypoints, speeds = self._compute_future_waypoints(
            sample_token, ego_trans_cur, ego_R_cur
        )
        # waypoints: (N, 2) x,y;  speeds: (N,) scalar speed
        waypoints_gt = np.column_stack([waypoints, speeds])  # (N, 3)

        # ---- Ego speed (from current → next) ---------------------------------
        ego_speed = float(speeds[0]) if len(speeds) > 0 else 0.0

        # ---- Obstacles -------------------------------------------------------
        obstacles = self._get_obstacles(
            sample_token, ego_trans_cur, ego_R_cur
        )

        # ---- Prompt ----------------------------------------------------------
        obstacle_summary = self._format_obstacle_summary(obstacles)
        prompt = self._PROMPT_TEMPLATE.format(
            speed=ego_speed, obstacle_summary=obstacle_summary
        )

        # ---- Target text (CoT rationale + waypoints) -------------------------
        target_text = self._build_target_text(
            scene_rec, obstacles, waypoints_gt
        )

        return {
            "image": image,
            "prompt": prompt,
            "target_text": target_text,
            "waypoints_gt": waypoints_gt,
            "obstacles": obstacles,
            "ego_speed": ego_speed,
            "sample_token": sample_token,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_table(path: Path) -> List[Dict]:
        """Load a nuScenes JSON table from disk.

        Args:
            path: Path to the JSON file.

        Returns:
            List of record dictionaries.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"nuScenes table not found: {path}")
        with open(path, "r") as fh:
            records: List[Dict] = json.load(fh)
        logger.debug("Loaded %d records from %s", len(records), path.name)
        return records

    def _load_image(self, sample_token: str) -> Image.Image:
        """Load and resize the CAM_FRONT image for a given sample.

        Args:
            sample_token: nuScenes sample token.

        Returns:
            Resized PIL Image in RGB mode.
        """
        sd = self._cam_sd_by_sample[sample_token]
        img_path = self._nuscenes_root / sd["filename"]
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert("RGB")
        # image_size is (H, W); PIL resize expects (W, H)
        target_h, target_w = self._image_size
        img = img.resize((target_w, target_h), Image.BILINEAR)
        return img

    def _get_ego_pose_for_sample(self, sample_token: str) -> Dict:
        """Retrieve the ego pose associated with a sample's CAM_FRONT data.

        Args:
            sample_token: nuScenes sample token.

        Returns:
            Ego-pose record dict with ``translation`` and ``rotation``.
        """
        sd = self._cam_sd_by_sample[sample_token]
        return self._ego_pose_by_token[sd["ego_pose_token"]]

    def _compute_future_waypoints(
        self,
        sample_token: str,
        ego_trans_cur: np.ndarray,
        ego_R_cur: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute future waypoints in ego-local BEV frame.

        Walks the ``next`` pointer chain from *sample_token* to gather
        ``num_future_waypoints`` successive ego poses, transforms them to the
        current ego frame, and estimates speed from consecutive positions.

        Args:
            sample_token: Current sample token.
            ego_trans_cur: (3,) current ego translation (global).
            ego_R_cur: (3, 3) current ego rotation matrix (ego-local → global).

        Returns:
            Tuple of:
              - waypoints: (N, 2) array of (x, y) in ego-local frame.
              - speeds: (N,) array of estimated speed at each waypoint (m/s).
        """
        future_translations: List[List[float]] = []
        cur_token = sample_token
        for _ in range(self._num_future_wps):
            cur_sample = self._sample_by_token.get(cur_token)
            if not cur_sample:
                break
            next_token = cur_sample.get("next", "")
            if not next_token or next_token not in self._cam_sd_by_sample:
                break # it means we reached the end of the log

            ego_pose = self._get_ego_pose_for_sample(next_token)
            future_translations.append(ego_pose["translation"])
            cur_token = next_token
        if not future_translations:
            return np.zeros((0,2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        points_global = np.array(future_translations, dtype=np.float64)
        local_positions = _global_to_ego(points_global, ego_trans_cur, ego_R_cur)
        waypoints_xy = local_positions[:,:2]
        dt = 1.0/ self._wp_freq_hz
        speeds = np.zeros(len(waypoints_xy), dtype=np.float64)

        speeds[0] = np.linalg.norm(waypoints_xy[0]) / dt
        for i in range(1, len(waypoints_xy)):
            speeds[i] = np.linalg.norm(waypoints_xy[i] - waypoints_xy[i-1]) / dt

        return waypoints_xy, speeds
            

    def _get_obstacles(
        self,
        sample_token: str,
        ego_trans_cur: np.ndarray,
        ego_R_cur: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Gather nearby obstacles from sample annotations in ego-local frame.

        Only obstacles within ``obstacle_radius_m`` of the ego vehicle are
        included.

        Args:
            sample_token: Current sample token.
            ego_trans_cur: (3,) ego translation (global).
            ego_R_cur: (3, 3) ego rotation matrix (ego-local → global).

        Returns:
            List of obstacle dicts, each with:
              - ``center``: [x, y] in ego-local frame.
              - ``size``: [w, l] (width, length) in metres.
              - ``yaw``: heading in ego-local frame (rad).
              - ``category``: human-readable category string.
        """
        anns = self._anns_by_sample.get(sample_token, [])
        ego_rot_cur = self._get_ego_pose_for_sample(sample_token)["rotation"]
        ego_yaw_global = _yaw_from_quaternion(*ego_rot_cur)

        obstacles: List[Dict[str, Any]] = []

        for ann in anns:
            center_global = np.array(ann["translation"], dtype=np.float64)
            #checking bev distance from ego vehicle
            dist = float(np.linalg.norm(center_global[:2] - ego_trans_cur[:2]))
            if dist > self._obstacle_radius:
                continue
            center_local_3d = _global_to_ego(center_global.reshape(1,3), ego_trans_cur, ego_R_cur)[0]
            center_xy = [float(center_local_3d[0]), float(center_local_3d[1])]

            obs_yaw_global = _yaw_from_quaternion(*ann["rotation"])
            obs_yaw_local = float(obs_yaw_global - ego_yaw_global)

            category = self._resolve_category(ann)

            size_wl = [float(ann["size"][0]), float(ann["size"][1])]

            obstacles.append({
                "center": center_xy,
                "size": size_wl,
                "yaw": obs_yaw_local,
                "category": category
            })

        return obstacles



    def _resolve_category(self, ann: Dict) -> str:
        """Resolve the human-readable category name for an annotation.

        Args:
            ann: Annotation record dict.

        Returns:
            Category name string (e.g. ``'human.pedestrian.adult'``).
        """
        instance_token = ann.get("instance_token", "")
        instance = self._instance_by_token.get(instance_token)
        if instance is None:
            return "unknown"
        cat_token = instance.get("category_token", "")
        category = self._category_by_token.get(cat_token)
        if category is None:
            return "unknown"
        return category.get("name", "unknown")

    # ------------------------------------------------------------------
    # Text generation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_obstacle_summary(obstacles: List[Dict[str, Any]]) -> str:
        """Create a concise natural-language summary of nearby obstacles.

        Args:
            obstacles: List of obstacle dicts from ``_get_obstacles``.

        Returns:
            Human-readable obstacle summary string.
        """
        if not obstacles:
            return "none detected"

        # Group obstacles by high-level category
        category_counts: Dict[str, int] = {}
        for obs in obstacles:
            # Use the top-level category (e.g. "vehicle" from "vehicle.car")
            cat = obs["category"]
            top_level = cat.split(".")[0] if "." in cat else cat
            category_counts[top_level] = category_counts.get(top_level, 0) + 1

        parts: List[str] = []
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            if count == 1:
                parts.append(f"1 {cat}")
            else:
                parts.append(f"{count} {cat}s")

        return ", ".join(parts)

    @staticmethod
    def _build_target_text(
        scene_rec: Dict,
        obstacles: List[Dict[str, Any]],
        waypoints_gt: np.ndarray,
    ) -> str:
        """Build the target output text with CoT rationale and waypoints.

        The rationale is synthesised from the scene description and obstacle
        information to provide a plausible chain-of-thought reasoning example
        for supervised fine-tuning.

        Args:
            scene_rec: nuScenes scene record.
            obstacles: List of obstacle dicts.
            waypoints_gt: (N, 3) array of (x, y, v) ground-truth waypoints.

        Returns:
            Target text string with ``<rationale>`` and ``<waypoints>`` tags.
        """

        desc = scene_rec.get("description", "Standard driving scenario").strip()
        if desc and not desc.endswith("."):
            desc += "."

        if obstacles:
            nearest = min(
                obstacles,
                key = lambda o: math.hypot(o["center"][0], o["center"][1])
            )
            dist = math.hypot(nearest["center"][0], nearest["center"][1])
            cat = nearest["category"].split(".")[0]
            direction = "ahead" if nearest["center"][0] >= 0 else "behind"
            obs_text = f"Nearest obstacle is {cat} at {dist:.1f}m {direction}."
        else:
            obs_text = "No nearby obstacles within safety radius"

        if len(waypoints_gt) > 0:
            final_y = float(waypoints_gt[-1,1])
            avg_speed = float(np.mean(waypoints_gt[:, 2]))

            if final_y > 1.5:
                lat_intent = "Steering left."
            elif final_y < -1.5:
                lat_intent = "Steering right"
            else:
                lat_intent = "Maintaining current lane."

            if avg_speed < 0.5: 
                log_intent = "Stopped or yielding"
            elif avg_speed < 4.0:
                lon_intent = "Moving at low speed."
            else: 
                lon_intent = f"Cruising at {avg_speed:.1f} m/s."

            intent_text = f"{lat_intent} {lon_intent}"

        else:
            intent_text = "Maintaining current state."

        rationale = f"Scene context: {desc} {obs_text} {intent_text}"

        wp_list = [
            [round(float(w[0]), 2), round(float(w[1]), 2), round(float(w[2]), 2)]
            for w in waypoints_gt
        ]
        wp_json = json.dumps(wp_list)

        return f"<rationale>{rationale}</rationale>\n<waypoints>{wp_json}</waypoints>"
    
    # ------------------------------------------------------------------
    # Public utilities
    # ------------------------------------------------------------------

    @property
    def valid_sample_tokens(self) -> List[str]:
        """Return the list of valid sample tokens in this split."""
        return [tok for tok, _ in self._valid_items]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for NuScenesVLADataset.

    PIL images and variable-length obstacle lists are kept as plain lists.
    Waypoints are stacked into a single tensor.  Scalar values are batched
    as tensors.

    Args:
        batch: List of sample dicts from ``__getitem__``.

    Returns:
        Collated batch dictionary.
    """
    return {
        "image": [item["image"] for item in batch],
        "prompt": [item["prompt"] for item in batch],
        "target_text": [item["target_text"] for item in batch],
        "waypoints_gt": torch.from_numpy(
            np.stack([item["waypoints_gt"] for item in batch], axis=0)
        ).float(),
        "obstacles": [item["obstacles"] for item in batch],
        "ego_speed": torch.tensor(
            [item["ego_speed"] for item in batch], dtype=torch.float32
        ),
        "sample_token": [item["sample_token"] for item in batch],
    }


def create_dataloaders(
    config: Dict[str, Any],
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders for NuScenesVLADataset.

    Args:
        config: Parsed configuration dictionary.
        batch_size: Number of samples per batch.
        num_workers: Number of parallel data-loading workers.

    Returns:
        Tuple of ``(train_loader, val_loader)``.
    """
    train_ds = NuScenesVLADataset(config, split="train")
    val_ds = NuScenesVLADataset(config, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    logger.info(
        "DataLoaders created — train: %d samples (%d batches), val: %d samples (%d batches)",
        len(train_ds),
        len(train_loader),
        len(val_ds),
        len(val_loader),
    )

    return train_loader, val_loader
