import os
import cv2
import h5py
import numpy as np
from einops import rearrange
from typing import List, Dict
from .perception import Frame, PinholeCamera, D435_COLOR_VFOV_DEG
from .vid_dec import decode_video_frames_torchcodec


def jpeg_encode(image: np.ndarray):
    return np.frombuffer(cv2.imencode(".jpg", image)[1].data, dtype=np.uint8)


def jpeg_decode(array: np.ndarray):
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


_WARNED_IDENTITY_EXTRINSICS = set()
_WARNED_DEFAULT_INTRINSICS = set()


def identity_extrinsics(num_frames: int):
    """(T, 4, 4) stack of identity poses, the no-extrinsics fallback.

    Identity is the only safe filler here: `ContextEncoder` rebases every camera onto
    camera 0 and then feeds the result to PRoPE, and `space_ee2cam` inverts it -- a
    zero matrix would produce NaN on the first `torch.inverse`. With identity
    everywhere PRoPE degenerates to a no-op (it is exactly the value camera 0 already
    takes after rebasing) and the action frame degenerates from camera-relative to
    base-relative, which is a valid, self-consistent representation.
    """
    return np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1))


def is_identity_extrinsics(wcT, atol: float = 1e-6):
    """True if every pose in `wcT` (..., 4, 4) is identity, i.e. the dataset carries no
    real extrinsics. Callers that project 3D points into the image use this to skip the
    overlay instead of drawing a geometrically meaningless one."""
    if wcT is None:
        return True
    if isinstance(wcT, np.ndarray):
        return bool(np.allclose(wcT, np.eye(4), atol=atol))
    import torch
    return bool(torch.allclose(wcT.float().cpu(), torch.eye(4), atol=atol))


def _warn_identity_extrinsics(source: str):
    """Warn once per source. A silent geometry fallback is the kind of thing that
    trains fine and then makes every projection/visualisation quietly wrong."""
    if source not in _WARNED_IDENTITY_EXTRINSICS:
        _WARNED_IDENTITY_EXTRINSICS.add(source)
        print("[WARN] no camera extrinsics found in {}, falling back to identity. "
              "PRoPE becomes a no-op and actions are expressed in the robot base "
              "frame instead of the camera frame; any wcT-based projection or "
              "trajectory overlay is meaningless.".format(source))


def default_intrinsics(height: int, width: int):
    """(3, 3) nominal RealSense D435 color K, the no-intrinsics fallback.

    `gen_norm_xy_map` turns K into the per-pixel normalized-plane coordinates that feed
    `ContextEncoder.proj_pe`, so a missing K cannot simply be skipped -- there is no
    "identity" for intrinsics the way there is for extrinsics. A fabricated-but-fixed K
    is fine there: the model only ever sees a constant coordinate grid and learns it as
    a 2D positional encoding. What it is NOT fine for is anything metric -- projecting
    3D points into the image, unprojecting depth, or comparing scale across cameras.
    Two cameras with genuinely different FOVs both getting this K makes them look
    identical to the geometry, so give real per-camera values as soon as you have them.
    """
    return PinholeCamera.realsense_d435(width=width, height=height).K.astype(np.float32)


def _warn_default_intrinsics(source: str):
    """Warn once per source, same reasoning as `_warn_identity_extrinsics`."""
    if source not in _WARNED_DEFAULT_INTRINSICS:
        _WARNED_DEFAULT_INTRINSICS.add(source)
        print("[WARN] no camera intrinsics found in {}, falling back to nominal "
              "RealSense D435 color K (vfov={} deg). Fine as a positional encoding, "
              "but every metric use -- projection, unprojection, cross-camera FOV -- "
              "is fabricated.".format(source, D435_COLOR_VFOV_DEG))


def gather_frames(
    traj: List[dict],
    cam_name: str, 
    indices,
    compress: bool,
):
    rgbs = []
    cam_poses = []

    for i in indices:
        cam = Frame.from_dict(traj[i][cam_name])

        color = cam.color
        if "compress" in cam.encoding:
            color = jpeg_decode(color)
        
        color = color[:, :, :3]  # remove alpha channel
        
        if "bgr" in cam.encoding:
            color = color[:, :, [2, 1, 0]]

        rgb = np.ascontiguousarray(color)
        if rgb.dtype == np.float32:
            rgb = (rgb * 255).astype(np.uint8)
        
        if compress:
            rgb = jpeg_encode(rgb)
        else:
            rgb = rearrange(rgb, "h w c -> c h w")

        rgbs.append(rgb)
        cam_poses.append(None if cam.wcT is None else np.asarray(cam.wcT))

    if not compress:
        rgbs = np.stack(rgbs, axis=0)  # (T, H, W, 3)

    if any(p is None for p in cam_poses):
        # all-or-nothing: a half-filled pose sequence is a bug in the converter, not a
        # dataset without extrinsics
        assert all(p is None for p in cam_poses), \
            "camera '{}' has extrinsics on some frames but not others".format(cam_name)
        _warn_identity_extrinsics("frame dict, camera '{}'".format(cam_name))
        cam_poses = identity_extrinsics(len(cam_poses))
    else:
        cam_poses = np.stack(cam_poses, axis=0)  # (T, 4, 4)

    camera = getattr(cam, "camera", None)
    if camera is None:
        # `cam.color` is still (H, W, C) -- `rgbs` may be jpeg bytes when compress=True
        H, W = cam.color.shape[:2]
        _warn_default_intrinsics("frame dict, camera '{}'".format(cam_name))
        K = default_intrinsics(H, W)
    else:
        K = np.asarray(camera.K, dtype=np.float32)

    outputs = {
        "rgb": rgbs,
        "K": K,
        "pose": cam_poses,
    }

    return outputs


def gather_ee_poses(traj: List[dict], indices):
    ee_poses = []
    for i in indices:
        ee_pose = traj[i]["ee_pose"]
        ee_poses.append(ee_pose)
    ee_poses = np.stack(ee_poses, axis=0)  # (T, nee, 4, 4)
    return ee_poses


def gather_grippers(traj: List[dict], indices):
    gripper = []
    for i in indices:
        width = traj[i]["gripper"]
        gripper.append(width)
    gripper = np.asarray(gripper)  # (T, nee)
    return gripper


def gather_states(traj: List[dict], indices):
    ee_poses = gather_ee_poses(traj, indices).astype(np.float32)
    gripper = gather_grippers(traj, indices).astype(np.float32)
    return compose_ee_gripper(ee_poses, gripper)


def gather_timestamps(traj: List[dict], indices):
    timestamps = np.array([traj[i]["timestamp"] for i in indices])
    return timestamps.astype(np.float32)


def compose_ee_gripper(ee_poses: np.ndarray, grippers: np.ndarray):
    T = ee_poses.shape[0]
    assert T == grippers.shape[0]
    aux_shape = ee_poses.shape[:-2]
    states = np.concatenate([ee_poses.reshape(*aux_shape, 16), 
                             grippers.reshape(*aux_shape, 1)], axis=-1)
    return states


def traj2dict(
    traj_data: List[Dict[str, np.ndarray]], 
    camera_names: List[str],
    prompt_text: str,
    compress: bool
):
    traj_len = len(traj_data)
    all_indices = np.arange(traj_len)

    flat_data_dict = {}

    for cam_name in camera_names:
        cam_frame_dict = gather_frames(traj_data, cam_name, all_indices, compress)
        flat_data_dict.update({f"{cam_name}/{k}": v for k, v in cam_frame_dict.items()})
    
    flat_data_dict["ee_pose"] = gather_ee_poses(traj_data, all_indices)
    flat_data_dict["gripper"] = gather_grippers(traj_data, all_indices)
    flat_data_dict["timestamp"] = gather_timestamps(traj_data, all_indices)
    
    attr_dict = {
        "prompt_text": prompt_text,
        "compress": compress
    }

    return flat_data_dict, attr_dict


def save_to_h5(path: str, data_dict: dict, attr_dict: dict):
    """
    data_dict:
    - ee_pose: np.ndarray of shape (T, 4, 4)
    - gripper: np.ndarray of shape (T,)
    - CAMERA_NAME_0:
        - rgb: np.ndarray of shape (T, 3, H, W) or list of vlen
        - pose: np.ndarray of shape (4, 4)
        - K: np.ndarray of shape (3, 3), camera intrinsic
        - time: np.ndarray of shape (T,)
    - CAMERA_NAME_1:
        - rgb: np.ndarray of shape (T, 3, H, W) or list of vlen
        - ...
    
    attr_dict:
    - prompt_text: str
    - compress: bool
    """
    output_dir = os.path.dirname(path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(path, "w") as h5:
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                h5.create_dataset(k, data=v)
            else:
                dtype = h5py.vlen_dtype(v[0].dtype)
                dset = h5.create_dataset(k, shape=(len(v),), dtype=dtype)
                for i, vi in enumerate(v):
                    dset[i] = vi
        
        for k, v in attr_dict.items():
            h5.attrs[k] = v


def slice_encoded_frames(
    camera_group: h5py.Group, 
    indices: np.ndarray,
    timestamp: np.ndarray = None, 
    video_root: str = None,
    skip_rgb: bool = False
):
    """Sample one camera group at `indices`.

    `skip_rgb` returns a 1x1 placeholder instead of decoding the frames, and is only for
    callers that need the geometry (`pose`, `K`) and not the pixels -- notably
    `data_prepare/compute_action_stats.py`, where jpeg/video decoding is ~all the cost
    and none of the answer. Everything else about the sampling stays identical, which is
    the point: the statistics must describe exactly the windows training will see.
    """

    sampled: Dict[str, np.ndarray] = {}
    if "K" in camera_group:
        sampled["K"] = camera_group["K"][:]  # (3, 3)
        if sampled["K"].ndim == 3:
            sampled["K"] = sampled["K"][0]
    # else: filled in after the loop, once the decoded image size is known

    for k in camera_group.keys():
        if k == "K":
            continue
        
        dset: h5py.Dataset = camera_group[k]
        
        if dset.dtype == np.object_:
            # compressed via jpeg encoding
            ind_clipped = np.clip(indices, 0, dset.len() - 1)

            if skip_rgb:
                sampled[k] = np.zeros((len(indices), 3, 1, 1), dtype=np.float32)
                continue

            first_sample = dset[ind_clipped[0]]

            if isinstance(first_sample, (str, bytes)):
                if isinstance(first_sample, bytes):
                    first_sample = first_sample.decode("utf-8")
                video_path = os.path.join(video_root, first_sample) if video_root else first_sample
                vid_ind = np.clip(indices, 0, len(timestamp) - 1)
                frames = decode_video_frames_torchcodec(
                    video_path=video_path,
                    timestamps=timestamp[vid_ind].tolist(),
                    tolerance_s=1e-2,
                    device="cpu"
                ).cpu().numpy()  # (N, C, H, W)
                sampled[k] = frames
            else:
                imgs_raw = [first_sample] + [dset[i] for i in ind_clipped[1:]]
                # imgs = [jpeg_decode(dset[i]) for i in ind_clipped]
                imgs = [jpeg_decode(raw) for raw in imgs_raw]
                imgs = [rearrange(img, "h w c -> c h w") for img in imgs]
                imgs = np.stack(imgs, axis=0)
                sampled[k] = imgs
        else:
            # raw data format
            # sampled[k] = dset[ind_clipped]
            sampled[k] = slice_dset(dset, indices)

    if "K" not in sampled:
        # decoded frames are (T, C, H, W) here, whatever the on-disk encoding was
        H, W = sampled["rgb"].shape[-2:]
        _warn_default_intrinsics("h5 group '{}'".format(camera_group.name))
        sampled["K"] = default_intrinsics(H, W)

    if "pose" not in sampled:
        _warn_identity_extrinsics("h5 group '{}'".format(camera_group.name))
        sampled["pose"] = identity_extrinsics(len(indices))

    return sampled


# def slice_dset(dset: h5py.Dataset, indices: np.ndarray):
#     indices = np.clip(indices, 0, dset.len() - 1)
#     return dset[indices]


def slice_dset(dset: h5py.Dataset, indices: np.ndarray):
    indices = np.clip(indices, 0, dset.len() - 1)
    return np.stack([dset[i] for i in indices], axis=0)

