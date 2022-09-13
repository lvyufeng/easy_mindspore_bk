import json
import mindspore.ops as ops
from mindspore import ms_class, context
from mindspore.train._utils import _exec_datagraph, _get_types_and_shapes
from mindspore.dataset.engine.offload import get_col_idxs, GetModelFromJson2Col
from mindspore import log as logger

@ms_class
class DataSinker():
    def __init__(self, dataset, steps, sink_size=-1, dynamic_shape=False, offload=False):
        self.dataset = dataset
        self.dataset_size = dataset.get_dataset_size()
        if sink_size <= self.dataset_size and self.dataset_size % sink_size != 0:
            raise ValueError(f"Dataset size {self.dataset_size} should be divisiable by 'sink_size'.")
        self.steps = steps
        self.sink_size = sink_size if sink_size == -1 else self.dataset_size
        self.offload = offload
        self.sink_count = steps // sink_size
        if steps % sink_size != 0:
            logger.warning(f"Steps number: {steps} is not divisiable by 'size_size', "
                           f"the remained steps will be ignored.")

        # transfer_dataset
        create_data_info_queue = (sink_size == 1 and self.dataset_size != 1 and
                                  context.get_context('device_target') == 'Ascend' and not dynamic_shape)
        self.transfer_dataset = _exec_datagraph(self.dataset, self.sink_size,
                                           create_data_info_queue=create_data_info_queue,
                                           is_dynamic_shape=dynamic_shape)
        # send data
        self.transfer_dataset.send(self.sink_count)

        # offload
        if offload:
            self.transform_list = []
            ds_cols = self.transfer_dataset.column_name
            json_offload = json.load(self.transfer_dataset._to_device.GetOffload())
            if json_offload is not None:
                for node in json_offload:
                    if node["op_type"] == "Map":
                        ds_col_ids = get_col_idxs(node["input_colums"], ds_cols)
                        self.transform_list.append(GetModelFromJson2Col(node, ds_col_ids))
            self.transform_list.reverse()

        # instantial GetNext op
        dataset_types, dataset_shapes = _get_types_and_shapes(dataset)
        queue_name = self.transfer_dataset.queue_name
        self.get_next = ops.GetNext(dataset_types, dataset_shapes, len(dataset_types), queue_name)

    def get_next(self):
        outputs = self.get_next()
        if self.offload:
            for transform in self.transform_list:
                outputs = transform(outputs)
        
        return outputs

    @property
    def sink_size(self):
        if self.sink_size > 0:
            return self.sink_size
        else:
            return self.dataset.get_dataset_size()
    
    def stop_send(self):
        self.transfer_dataset.stop_send()

    def release(self):
        self.transfer_dataset.release()
    
    def continue_send(self):
        self.transfer_dataset.continue_send()
    
    def reset(self):
        self.transfer_dataset._reset()
    
    def get_data_info(self):
        return self.transfer_dataset.get_data_info()

    def dynamic_min_max_shapes(self):
        return self.dataset.dynamic_min_max_shapes()

def data_sink(fn=None, dataset=None, steps=1, sink_size=-1, dynamic_shape=False, offload=False):
    data_sinker = DataSinker(steps, sink_size, dynamic_shape, offload)
    def inner():
        pass
    return inner