import os

from .ObjMesh import ObjMesh
from utils.io import read_json_file


class BOPObjSet:
    def __init__(self, obj_list, path, device=None):
        self.obj_list = obj_list
        self.path = path
        self.device = device if device is not None else 'cpu'
        path_models = os.path.join(path, 'models')
        path_models_eval = os.path.join(path, 'models_eval')
        objects_info = read_json_file(os.path.join(path_models, 'models_info.json'))
        objects_info_eval = read_json_file(os.path.join(path_models_eval, 'models_info.json'))
        self.objects = {}
        self.objects_eval = {}
        for obj_id in obj_list:
            self.objects[obj_id] = ObjMesh(device=self.device, obj_id=int(obj_id), name=obj_list[obj_id], is_eval=False,
                                           mesh_path=os.path.join(path_models, f'obj_{int(obj_id):0>6d}.ply'),
                                           **objects_info[str(obj_id)])
            self.objects_eval[obj_id] = ObjMesh(device=self.device, obj_id=int(obj_id), name=obj_list[obj_id],
                                                is_eval=True,
                                                mesh_path=os.path.join(path_models_eval, f'obj_{int(obj_id):0>6d}.ply'),
                                                **objects_info_eval[str(obj_id)])
