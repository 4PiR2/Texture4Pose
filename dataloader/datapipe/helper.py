from typing import Any, Callable, Iterator, Sequence, Union

import torch
import torchdata
from torch.utils.data import functional_datapipe
from torch.utils.data.dataset import T_co

from config import const as cc
from dataloader.sample import Sample


class IterDataPipe(torchdata.datapipes.iter.IterDataPipe[T_co]):
    def __init__(self, *source_datapipes: torchdata.datapipes.iter.IterDataPipe, required_attributes: list[str] = None,
                 **kwargs):
        self.source_datapipes: tuple[torchdata.datapipes.iter.IterDataPipe] = source_datapipes
        for key, value in kwargs.items():
            self.__setattr__(key, value)

        if required_attributes is not None:
            if isinstance(required_attributes, str):
                required_attributes = [required_attributes]
        else:
            required_attributes = []
        self.required_attributes: list[str] = required_attributes

        unavailable_keys = []
        for key in self.required_attributes:
            if not hasattr(self, key):
                unavailable_keys.append(key)
        assert len(unavailable_keys) == 0

    def __getattr__(self, attribute_name: str):
        try:
            return super().__getattr__(attribute_name)
        except AttributeError as _:
            if not attribute_name.startswith('_'):
                for dp in self.source_datapipes:
                    try:
                        return getattr(dp, attribute_name)
                    except AttributeError as _:
                        pass
            raise AttributeError

    def __iter__(self) -> Iterator[T_co]:
        if self.source_datapipes:
            yield from torchdata.datapipes.iter.Zipper(*self.source_datapipes).map(lambda x: self.map_fn(*x))
        else:
            while True:
                yield self.map_fn()

    def map_fn(self, *data) -> T_co:
        raise NotImplementedError


class SampleSource(IterDataPipe[Sample]):
    def __init__(self, dtype: torch.dtype = cc.dtype, device: torch.device = cc.device, scene_mode: bool = True,
                 img_render_size: int = 512, **kwargs):
        super().__init__(**{**dict(
            dtype=dtype, device=device, scene_mode=scene_mode, img_render_size=img_render_size,
            objects={}, objects_eval={},
        ), **kwargs})

    def map_fn(self):
        return Sample()

    @property
    def valid_args(self) -> set[str]:
        return {'N'}


@functional_datapipe('map_sample')
class SampleMapperIDP(IterDataPipe[Sample]):
    def __init__(self, src_dp: IterDataPipe, request_args: list[str] = None, response_args: list[str] = None,
                 delete_args: list[str] = None, fn: Callable[[Any], Union[Sequence[torch.Tensor], torch.Tensor]] = None,
                 *other_dps: IterDataPipe, required_attributes: list[str] = None):
        assert len(set(request_args) - src_dp.valid_args) == 0  # sanity check
        super().__init__(src_dp, *other_dps, required_attributes=required_attributes)
        self._fn: Callable = fn if fn is not None else self.main
        self._request_args: list[str] = request_args if request_args is not None else []
        self._response_args: list[str] = response_args if response_args is not None else []
        self._delete_args: list[str] = delete_args if delete_args is not None else []

    def map_fn(self, sample: Sample, *other_data) -> Sample:
        response = self._fn(*other_data, **{req_arg: getattr(sample, req_arg) for req_arg in self._request_args})
        if len(self._response_args) == 1:
            response = (response,)
        assert len(self._response_args) == len(response)
        for resp_arg, resp in zip(self._response_args, response):
            if resp_arg == '_':
                continue
            setattr(sample, resp_arg, resp)
        for del_arg in self._delete_args:
            if hasattr(sample, del_arg):
                delattr(sample, del_arg)
        return sample

    def main(self, *args, **kwargs) -> Union[Sequence[torch.Tensor], torch.Tensor, Any]:
        if args or kwargs:
            raise NotImplementedError
        return None

    @property
    def valid_args(self) -> set[str]:
        valid_args = set(self._response_args)
        try:
            valid_args = valid_args.union(self.source_datapipes[0].valid_args)
        except AttributeError as _:
            pass
        valid_args -= set(self._delete_args)
        return valid_args


@functional_datapipe('batch_sample')
class SampleBatcherIDP(IterDataPipe[Sample]):
    def __init__(self, src_dp: torchdata.datapipes.iter.IterDataPipe, batch_size: int = 1, unbatch_src: bool = False):
        super().__init__(src_dp)
        self._batch_size: int = batch_size
        self._unbatch_src: bool = unbatch_src

    def __iter__(self) -> Sample:
        dp = self.source_datapipes[0]
        if self._unbatch_src:
            dp = dp.unbatch_sample()
        batch = []
        for sample in dp:
            batch.append(sample)
            if len(batch) == self._batch_size:
                out = Sample(*batch)
                batch = []
                yield out


@functional_datapipe('unbatch_sample')
class UnSampleBatcherIDP(IterDataPipe[Sample]):
    def __init__(self, src_dp: torchdata.datapipes.iter.IterDataPipe):
        super().__init__(src_dp)

    def __iter__(self) -> Sample:
        for sample in self.source_datapipes[0]:
            yield from sample


@functional_datapipe('filter_sample')
class SampleFiltererIDP(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, request_args: list[str] = None, delete_args: list[str] = None,
                 fn: Callable[[Any], torch.Tensor] = None, *other_dps: IterDataPipe,
                 required_attributes: list[str] = None):
        super().__init__(src_dp, request_args, delete_args=delete_args, fn=fn, *other_dps,
                         required_attributes=required_attributes)

    def map_fn(self, sample: Sample, *other_data) -> Sample:
        mask: torch.Tensor = \
            self._fn(*other_data, **{req_arg: getattr(sample, req_arg) for req_arg in self._request_args})
        for del_arg in self._delete_args:
            if hasattr(sample, del_arg):
                delattr(sample, del_arg)
        for key, value in sample.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(sample, key, value[mask])
            elif isinstance(value, list):
                raise NotImplementedError
            else:
                setattr(sample, key, value)
        return sample


@functional_datapipe('repeat_sample')
class SampleRepeaterIDP(IterDataPipe[Sample]):
    def __init__(self, src_dp: torchdata.datapipes.iter.IterDataPipe, repeat: int = 1, batch: bool = False):
        super().__init__(src_dp)
        self._repeat: int = repeat
        self._batch: bool = batch

    def __iter__(self) -> Sample:
        dp = self.source_datapipes[0]
        for sample in dp:
            if self._repeat <= 0:
                yield sample
            elif self._batch:
                batch = [sample.clone() for _ in range(self._repeat)]
                yield Sample(*batch)
            else:
                for _ in range(self._repeat):
                    yield sample.clone()
