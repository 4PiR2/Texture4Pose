import os

import torch
import tqdm


if __name__ == '__main__':
    ckpt_path_list = [f'/home/user/project/Texture4Pose/outputs/detectron2_logs/version_{i}/model_final.pth' for i in range(2)]
    for ckpt_path in tqdm.tqdm(ckpt_path_list):
        state_dict = torch.load(os.path.join(ckpt_path))['model']
        for k, v in state_dict.items():
            state_dict[k] = v.to('cpu')
        torch.save({'model': state_dict}, ckpt_path.split('.')[0] + '_cpu.pth')


    ckpt_path_list = os.listdir('cuda')
    for ckpt_path in tqdm.tqdm(ckpt_path_list):
        state_dict = torch.load(os.path.join('cuda', ckpt_path))['state_dict']
        for k, v in state_dict.items():
            state_dict[k] = v.to('cpu')
        torch.save({'state_dict': state_dict}, ckpt_path)
