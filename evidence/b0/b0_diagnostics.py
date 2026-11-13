"""Read-only memory RPCs used only in separate profiler services."""
import dataclasses
import json
import os

import torch
from vllm_ascend.worker.worker import NPUWorker


class MemoryWorker(NPUWorker):
    def b0_memory_snapshot(self, reset_peak='false'):
        torch.npu.synchronize()
        if reset_peak=='true':torch.npu.reset_peak_memory_stats()
        runner=self.model_runner
        model=self.get_model()
        weights={}
        parameter_storages={}
        for name,tensor in model.named_parameters():
            if tensor.device.type=='npu':
                storage=tensor.untyped_storage()
                parameter_storages[(storage.data_ptr(),storage.nbytes())]=storage.nbytes()
        for name,tensor in list(model.named_parameters())+list(model.named_buffers()):
            if tensor.device.type!='npu':continue
            storage=tensor.untyped_storage()
            key=(storage.data_ptr(),storage.nbytes())
            weights.setdefault(key,{'bytes':storage.nbytes(),'names':[]})['names'].append(name)
        cfg=runner.kv_cache_config
        kinds={name:('state' if 'Mamba' in type(g.kv_cache_spec).__name__ else 'attention_kv')
               for g in cfg.kv_cache_groups for name in g.layer_names}
        allocations=[]
        for tensor in cfg.kv_cache_tensors:
            classes=sorted({kinds.get(n,'unknown') for n in tensor.shared_by})
            allocations.append({'bytes':tensor.size,'shared_by':tensor.shared_by,'classes':classes})
        logical=[]
        def tensors(value):
            if isinstance(value,torch.Tensor):yield value
            elif isinstance(value,(list,tuple)):
                for child in value:yield from tensors(child)
            elif isinstance(value,dict):
                for child in value.values():yield from tensors(child)
        for name,kind in kinds.items():
            module=runner.compilation_config.static_forward_context.get(name)
            arrays=list(tensors(getattr(module,'kv_cache',[])))
            logical.append({'layer':name,'kind':kind,'tensors':[
                {'shape':list(t.shape),'dtype':str(t.dtype),'logical_bytes':t.numel()*t.element_size(),
                 'storage_pointer':t.untyped_storage().data_ptr(),'storage_bytes':t.untyped_storage().nbytes(),
                 'storage_offset':t.storage_offset()} for t in arrays]})
        return {'pid':os.getpid(),'rank':self.rank,'logical_device':torch.npu.current_device(),
            'weight_and_model_buffer_storage_bytes':sum(x['bytes'] for x in weights.values()),
            'parameter_storage_bytes':sum(parameter_storages.values()),
            'model_buffer_storage_bytes_excluding_parameter_aliases':sum(
                x['bytes'] for key,x in weights.items() if key not in parameter_storages),
            'weight_storage_count':len(weights),'cache_allocations':allocations,
            'cache_num_blocks':cfg.num_blocks,
            'cache_physical_config_bytes':sum(t.size for t in cfg.kv_cache_tensors),
            'cache_groups':json.loads(json.dumps([dataclasses.asdict(g) for g in cfg.kv_cache_groups],default=str)),
            'cache_layer_logical_views':logical,
            'torch_allocated_bytes':torch.npu.memory_allocated(),
            'torch_reserved_bytes':torch.npu.memory_reserved(),
            'torch_max_allocated_bytes':torch.npu.max_memory_allocated(),
            'torch_max_reserved_bytes':torch.npu.max_memory_reserved(),
            'npu_graph_memory_bytes_worker_field':getattr(self,'npugraph_memory_bytes',None),
            'available_cache_budget_bytes':getattr(self,'available_kv_cache_memory_bytes',None),
            'profile_scope':'diagnostic service; not formal timing'}
