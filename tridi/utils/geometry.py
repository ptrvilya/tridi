from typing import Union

import torch
import torch.nn.functional as F
import numpy as np


def rotation_6d_to_matrix(d6: Union[np.ndarray, torch.Tensor]):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if isinstance(d6, np.ndarray):
        d6 = torch.from_numpy(d6)

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: Union[np.ndarray, torch.Tensor]):
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix)

    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def matrix_to_aa(matrix: Union[np.ndarray, torch.Tensor]):
    """
    Converts rotation matrix to axis-angle representation.

    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        aa: axis-angle representation, of size (*, 3)
    """
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix).float()
        is_numpy = True
    else:
        is_numpy = False

    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0)
        is_squeeze = True
    else:
        is_squeeze = False

    # Axes
    axis = torch.zeros(matrix.size()[:-2] +  (3,), dtype=torch.float, device=matrix.device)
    axis[..., 0] = matrix[..., 2, 1] - matrix[..., 1, 2]
    axis[..., 1] = matrix[..., 0, 2] - matrix[..., 2, 0]
    axis[..., 2] = matrix[..., 1, 0] - matrix[..., 0, 1]

    # Angle
    r = torch.hypot(axis[..., 0] + 1e-6, torch.hypot(axis[..., 1], axis[..., 2] + 1e-6))
    t = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    theta = torch.atan2(r, t - 1)

    # Normalise the axis
    axis = axis / (r.unsqueeze(-1) + 1e-5)

    aa = axis * theta.unsqueeze(-1)

    if is_squeeze:
        aa = aa.squeeze(0)

    if is_numpy:
        aa = aa.numpy()
    return aa


def rotation_6d_to_aa(d6: Union[np.ndarray, torch.Tensor]):
    """
    Converts 6D rotation representation by Zhou et al. [1] to axis-angle representation.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of axis-angle representations of size (*, 3)
    """
    if isinstance(d6, np.ndarray):
        d6 = torch.from_numpy(d6).float()
        is_numpy = True
    else:
        is_numpy = False

    matrix = rotation_6d_to_matrix(d6)
    aa = matrix_to_aa(matrix)

    if is_numpy:
        aa = aa.numpy()

    return aa


def batch_rodrigues(
    rot_vecs: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat