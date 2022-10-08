import cv2
import pyzbar.pyzbar
import torch


def decode_barcode(img: torch.Tensor, use_zbar: bool = True):
    if not use_zbar:
        bardet = cv2.barcode_BarcodeDetector()
    flag = False
    if img.ndim == 2:
        img = img[None, None]
        flag = True
    elif img.ndim == 3:
        img = img[None]
        flag = True
    img = (img * 255.).permute(0, 2, 3, 1).detach().cpu().numpy().astype('uint8')
    decoded = []
    for im in img:
        if use_zbar:
            ret = pyzbar.pyzbar.decode(im)
            decoded.append(ret[0].data.decode('UTF-8') if len(ret) else None)
        else:
            ret, decoded_info, decoded_type, corners = bardet.detectAndDecode(im)
            decoded.append(decoded_info[0] if ret else None)
    return decoded[0] if flag else decoded
