import json


def read_json_file(path):
    with open(path, 'r') as f:
        return json.load(f)

def parse_device(device=None):
    return device if device is not None else 'cpu'
