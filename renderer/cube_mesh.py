import torch
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
from pytorch3d.structures.meshes import Meshes


# sphere: #verts[i] = 10 * 4 ** i + 2, #faces[i] = 5 * 4 ** (i + 1)
# cube: #verts[i] = 6 * 4 ** i + 2, #faces[i] = 3 * 4 ** (i + 1)
# tetrahedron: #verts[i] = 2 * 4 ** i + 2, #faces[i] = 4 ** (i + 1)
# cylinderstrip: #verts[i] = 6 * 4 ** i + 2, #faces[i] = 2 * 4 ** (i + 1)
# sphericon: #verts[i] = 2 ** (i + 2) + 2, #faces[i] = 2 ** (i + 3)


def _get_mesh(verts0, faces0, level: int = 0, device: torch.device = None):
    if level < 0:
        raise ValueError('level must be >= 0.')
    if device is None:
        device = torch.device('cpu')
    verts = torch.tensor(verts0, dtype=torch.float32, device=device)
    faces = torch.tensor(faces0, dtype=torch.int64, device=device)
    mesh = Meshes(verts=[verts], faces=[faces])
    subdivide = SubdivideMeshes()
    for _ in range(level):
        mesh = subdivide(mesh)
    return mesh


def cube(level: int = 0, base: bool = True, device: torch.device = None):
    """
    Create verts and faces for a unit cube, with all faces oriented
    consistently.

    Args:
        level: integer specifying the number of iterations for subdivision
               of the mesh faces. Each additional level will result in four new
               faces per face.
        base: whether to have top and bottom surface
        device: A torch.device object on which the outputs will be allocated.

    Returns:
        Meshes object with verts and faces.
    """
    # Vertex coordinates for a level 0 cube
    verts0 = [
        [-1., -1., 1],
        [-1., 1., 1.],
        [-1., 1., -1.],
        [1., 1., 1.],
        [1., 1., -1.],
        [1., -1., 1.],
        [1., -1., -1.],
        [-1., -1., -1.],
    ]
    # Faces for level 0 cube
    faces0 = [
        [0, 1, 2],
        [1, 3, 4],
        [3, 5, 6],
        [0, 7, 5],
        [7, 2, 4],
        [5, 3, 1],
        [7, 0, 2],
        [2, 1, 4],
        [4, 3, 6],
        [5, 7, 6],
        [6, 7, 4],
        [0, 5, 1],
    ]
    if not base:
        faces0 = [[a, b, c] for a, b, c in faces0 if not verts0[a][-1] == verts0[b][-1] == verts0[c][-1]]
    return _get_mesh(verts0, faces0, level, device)


def tetrahedron(level: int = 0, device: torch.device = None):
    verts0 = [
        [-1., -1., 1],
        [-1., 1., -1.],
        [1., 1., 1.],
        [1., -1., -1.],
    ]
    faces0 = [
        [0, 2, 1],
        [0, 1, 3],
        [0, 3, 2],
        [1, 2, 3],
    ]
    return _get_mesh(verts0, faces0, level, device)


def cylinder_strip(level: int = 0, device: torch.device = None):
    # no base
    if level == 0:
        mesh = cube(0, False, device)
        verts = mesh.verts_list()[0]
        sin_45 = 2 ** .5 * .5
        rot = torch.tensor([
            [sin_45, sin_45, 0.],
            [-sin_45, sin_45, 0.],
            [0., 0., 1.],
        ], dtype=verts.dtype, device=verts.device)
        verts = verts @ rot
    else:
        subdivide = SubdivideMeshes()
        mesh = subdivide(cylinder_strip(level - 1, device))
        verts = mesh.verts_list()[0]
    verts[:, :2] /= verts[:, :2].norm(p=2, dim=1, keepdim=True)
    faces = mesh.faces_list()[0]
    return Meshes(verts=[verts], faces=[faces])


def sphericon(level: int = 0, device: torch.device = None) -> Meshes:
    verts0 = [
        [1., 0., 0.],
        [-1., 0., 0.],
        [0., 1., 0.],
        [0., -1., 0.],
        [0., 0., 1.],
        [0., 0., -1.],
    ]
    theta = torch.linspace(0., torch.pi * .5, 2 ** level + 1, device=device)[1:-1]
    cos_theta, sin_theta, zeros_theta = theta.cos(), theta.sin(), torch.zeros_like(theta)
    verts = torch.cat([
        torch.tensor(verts0, device=device),
        torch.stack([cos_theta, zeros_theta, sin_theta], dim=-1),
        torch.stack([-sin_theta, zeros_theta, cos_theta], dim=-1),
        torch.stack([zeros_theta, -cos_theta, -sin_theta], dim=-1),
        torch.stack([zeros_theta, sin_theta, -cos_theta], dim=-1),
    ], dim=0)
    if level <= 0:
        faces = [[2, 4, 0], [2, 1, 4], [1, 2, 5], [1, 5, 3], [3, 4, 1], [3, 0, 4], [0, 3, 5], [0, 5, 2]]  # octahedron
    else:
        faces = []
        faces += [[2, len(theta) * 0 + 6, 0]] + \
                 [[2, i + 1, i] for i in range(len(theta) * 0 + 6, len(theta) * 1 + 5)] + \
                 [[2, 4, len(theta) * 1 + 5]]
        faces += [[2, len(theta) * 1 + 6, 4]] + \
                 [[2, i + 1, i] for i in range(len(theta) * 1 + 6, len(theta) * 2 + 5)] + \
                 [[2, 1, len(theta) * 2 + 5]]
        faces += [[1, 2, len(theta) * 4 + 5]] + \
                 [[1, i + 1, i] for i in range(len(theta) * 3 + 6, len(theta) * 4 + 5)[::-1]] + \
                 [[1, len(theta) * 3 + 6, 5]]
        faces += [[1, 5, len(theta) * 3 + 5]] + \
                 [[1, i + 1, i] for i in range(len(theta) * 2 + 6, len(theta) * 3 + 5)[::-1]] + \
                 [[1, len(theta) * 2 + 6, 3]]
        faces += [[3, len(theta) * 2 + 5, 1]] + \
                 [[3, i, i + 1] for i in range(len(theta) * 1 + 6, len(theta) * 2 + 5)[::-1]] + \
                 [[3, 4, len(theta) * 1 + 6]]
        faces += [[3, len(theta) * 1 + 5, 4]] + \
                 [[3, i, i + 1] for i in range(len(theta) * 0 + 6, len(theta) * 1 + 5)[::-1]] + \
                 [[3, 0, len(theta) * 0 + 6]]
        faces += [[0, 3, len(theta) * 2 + 6]] + \
                 [[0, i, i + 1] for i in range(len(theta) * 2 + 6, len(theta) * 3 + 5)] + \
                 [[0, len(theta) * 3 + 5, 5]]
        faces += [[0, 5, len(theta) * 3 + 6]] + \
                 [[0, i, i + 1] for i in range(len(theta) * 3 + 6, len(theta) * 4 + 5)] + \
                 [[0, len(theta) * 4 + 5, 2]]
    faces = torch.tensor(faces, device=device)
    return Meshes(verts=[verts], faces=[faces])
