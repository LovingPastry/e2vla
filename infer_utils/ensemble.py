import torch
import numpy as np
import pypose as pp
from collections import deque
from data_utils.align import interp_linear, interp_SO3


class TrajEnsembler(object):
    """Temporal ensembling of overlapping action chunks (ACT-style).

    Every inference call emits a whole chunk of `action_horizon` future states, but
    only the first few are executed before the next chunk arrives. Successive chunks
    therefore overlap in time; averaging them removes the discontinuity at chunk
    boundaries.

    `history_traj` and `history_time` are two views of the SAME chunks and must stay
    index-aligned at all times -- `update()` pairs them positionally via zip(), so any
    length mismatch silently mixes one chunk's poses with another chunk's timestamps.
    """

    def __init__(self, max_hist_len=-1):
        maxlen = max_hist_len if max_hist_len > 0 else None
        self.max_hist_len = max_hist_len
        self.history_traj = deque(maxlen=maxlen)
        self.history_time = deque(maxlen=maxlen)

    def reset(self):
        """Drop all history. Must be called between episodes: timestamps restart at
        zero, so a chunk left over from the previous episode would be interpolated
        onto the new episode's query times and blended into its first actions."""
        self.history_traj.clear()
        self.history_time.clear()

    def _weighted_sum_linear(cls, trajs: np.ndarray, weights: np.ndarray):
        weights = weights.reshape(weights.shape + (1,) * (trajs.ndim - weights.ndim))  # (N, T, 1)
        ensembled_traj = (trajs * weights).sum(axis=0) / weights.sum(axis=0)
        return ensembled_traj

    def _weighted_sum_SO3(cls, trajs: np.ndarray, weights: np.ndarray):
        """
        Args:
            trajs (np.ndarray): (N, T, 3, 3)
            weights (np.ndarray): (N, T)
        
        Returns:
            ensembled_traj (np.ndarray): (T, 3, 3)
        """
        Log = pp.SO3_type.Log
        Exp = pp.so3_type.Exp
        Mul = pp.SO3_type.Mul
        Inv = pp.SO3_type.Inv
        
        trajs = pp.from_matrix(torch.from_numpy(trajs), pp.SO3_type)
        # anchor on the newest chunk, average the tangent-space deltas around it
        init_SO3 = trajs[-1]  # (T, 3, 3)
        delta_so3 = Log(Mul(Inv(init_SO3), trajs)).tensor()  # (N, T, 3)

        weights = weights.reshape(weights.shape + (1,) * (delta_so3.ndim - weights.ndim))  # (N, T, 1)
        weights = torch.from_numpy(weights).to(delta_so3)
        update_so3 = (delta_so3 * weights).sum(dim=0) / weights.sum(dim=0)
        
        ensembled_traj = Mul(init_SO3, Exp(update_so3)).matrix()
        return ensembled_traj.numpy()
    
    def update(self, new_traj: np.ndarray, new_time: np.ndarray, on_SO3: bool = False):
        """Blend the newest chunk with the still-overlapping previous chunks.

        Each previous chunk is resampled onto `new_time` by interpolation, then all
        candidates are averaged with weights that ramp from 0.1 (oldest) to 1.0
        (newest), so the freshest prediction dominates while older ones smooth it.

        Args:
            new_traj (np.ndarray): (T, D) or (T, D1, D2), the chunk just predicted
            new_time (np.ndarray): (T,), timestamps of `new_traj`
            on_SO3 (bool): average in the tangent space of SO(3) instead of linearly

        Returns:
            ensembled_traj (np.ndarray): same shape as new_traj
        """
        self.history_traj.append(new_traj)  # (nhist, T, ...)
        self.history_time.append(new_time)  # (nhist, T)
        assert len(self.history_traj) == len(self.history_time), (
            "traj/time history desynced ({} vs {}); they are zipped positionally below"
            .format(len(self.history_traj), len(self.history_time))
        )

        # drop chunks that no longer overlap the newest one -- nothing to interpolate
        while True:
            initial_end_time = self.history_time[0][-1]
            current_start_time = self.history_time[-1][0]
            if current_start_time > initial_end_time:
                self.history_traj.popleft()
                self.history_time.popleft()
            else:
                break

        candidate_trajs = []
        candidate_weights = []
        if len(self.history_traj) > 1:
            # [:-1] skips the chunk we just appended; it is added verbatim below
            for prev_traj, prev_time in zip(list(self.history_traj)[:-1],
                                            list(self.history_time)[:-1]):
                # only query times strictly inside prev_time's span can be interpolated
                bin_indices = np.digitize(new_time, prev_time)
                mask = (bin_indices > 0) & (bin_indices < len(prev_time))
                candidate_weights.append(mask.astype(np.float32))
                
                candidate_traj = new_traj.copy()
                l_ind = bin_indices[mask] - 1
                t0 = prev_time[l_ind]
                t1 = prev_time[l_ind + 1]
                
                q0 = prev_traj[l_ind]
                q1 = prev_traj[l_ind + 1]
                if mask.any():
                    if on_SO3:
                        candidate_traj[mask] = interp_SO3(q0, q1, t0, t1, new_time[mask])
                    else:
                        candidate_traj[mask] = interp_linear(q0, q1, t0, t1, new_time[mask])
                candidate_trajs.append(candidate_traj)
        
        candidate_trajs.append(new_traj)
        candidate_weights.append(np.ones(len(new_traj)))
        
        candidate_trajs = np.asarray(candidate_trajs)  # (N, T, D) or (N, T, D1, D2)
        candidate_weights = np.asarray(candidate_weights)  # (N, T)
        
        # set decay factor
        decay_factor = self.ensemble_weights(candidate_weights.shape[0])  # (N,)
        candidate_weights = candidate_weights * decay_factor[:, None]
        
        if on_SO3:
            ensembled_traj = self._weighted_sum_SO3(candidate_trajs, candidate_weights)
        else:
            ensembled_traj = self._weighted_sum_linear(candidate_trajs, candidate_weights)
        return ensembled_traj
    
    @property
    def num_history(self):
        return len(self.history_time)

    def ensemble_weights(self, N: int):
        """
        Args:
            N (int): number of history trajectory
        
        Returns:
            weight (np.ndarray): shape = (N,), value from small (earliest) to large (latest)
        """
        return np.linspace(0.1, 1, N)

