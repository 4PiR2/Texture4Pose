import pathlib
import numpy as np
import cv2
from tensorboard.backend.event_processing import event_accumulator


def main():
    record_path = 'outputs/lightning_logs/version_155/events.out.tfevents.1664761124.cnb-d102-58.143439.9'
    output_path = '/home/user/Desktop/tmp'

    event_acc = event_accumulator.EventAccumulator(
        record_path, size_guidance={'images': 0})
    event_acc.Reload()

    outdir = pathlib.Path(output_path)
    outdir.mkdir(exist_ok=True, parents=True)

    for tag in event_acc.Tags()['images']:
        events = event_acc.Images(tag)

        tag_name = tag.replace('/', '_')
        dirpath = outdir / tag_name
        dirpath.mkdir(exist_ok=True, parents=True)

        for index, event in enumerate(events):
            s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
            image = cv2.imdecode(s, cv2.IMREAD_COLOR)
            outpath = dirpath / '{:04}.jpg'.format(index)
            cv2.imwrite(outpath.as_posix(), image)


if __name__ == '__main__':
    main()
