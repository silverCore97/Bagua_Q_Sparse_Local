#!/usr/bin/env python3
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup
from torch.optim.optimizer import Optimizer
import torch
import math
import numpy as np
from typing import List, Tuple

# Creates compressed tensor and updates error compensation term
def qsl(eta_grad,
        memory,
        qsl_grad,
        k,
        topK_flag,
        s,      # Number quantization levels
        sparsify,
        use_memory,
        quantization_scheme,
        use_normalization
        ):
    
    # Sign quantization
    def signq(var):
        one_norm = torch.norm(var, p=1)
        return one_norm * torch.sign(var + 1e-13) / float(torch.numel(var))

    # QSGD quantization
    def qsgd(var):
        level_float = s * torch.abs(var) / norm1
        previous_level = torch.floor(level_float)
        is_next_level = (torch.rand(var.size(), dtype=torch.float32, device = 'cuda') < (level_float - previous_level))
        is_next_level = is_next_level.float()
        new_level = previous_level + is_next_level
        unnormalized = torch.sign(var) * new_level * norm1 / s
        beta = float(torch.numel(var)) / float(s * s)
        return unnormalized / (1.0 + beta) if use_normalization else unnormalized

    def get_quantization(q):
        if q == 'qsgd':
            return qsgd
        elif q == 'sign':
            return signq
        else:
            return lambda x: x

    if not sparsify:
        norm1 = torch.norm(eta_grad) + torch.constant(1e-5, dtype=torch.float32)
        if use_memory:
            input = memory + eta_grad
        else:
            input = eta_grad

        func = get_quantization(quantization_scheme)
        q = func(input)

        memory.mul_(0).add_(input-q)
        qsl_grad.mul_(0).add_(q)

    if use_memory:
        input = memory + eta_grad
    else:
        input = eta_grad

    org_shape = input.size()
    numel = torch.numel(input)
    K = min(numel, k)  

    # Choice of sparsification
    if topK_flag:
        _, indices = torch.topk(torch.reshape(torch.abs(input), [-1]), K)
    else:
        indices = torch.from_numpy(np.random.choice(torch.range(numel), K, False))


    flat_input = torch.reshape(input, [-1])
    values = torch.gather(flat_input, 0, indices)  # dim=0
    norm1 = torch.norm(values)
    quantization_func = get_quantization(quantization_scheme)
    flattened_quantized = torch.zeros_like(flat_input).scatter(0, indices,
                                                               quantization_func(values))
    quantization = torch.reshape(flattened_quantized, shape=org_shape)

    q_func = lambda: quantization
    zero_tensor = lambda: torch.zeros_like(input, dtype=torch.float32)

    if float(0) < norm1:
        q = q_func()
    else:
        q = zero_tensor()

    err = input - q

    memory.mul_(0).add_(err)
    qsl_grad.mul_(0).add_(q)
       



class QSparseLocalOptimizer(Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-3,  
            k: int = 1000,
            schedule: int = 1,
            quantization_scheme: str  = 'sign',
            sparsify: bool = True,
            use_memory: bool = True,
            quantization_levels: int = 256,
            top_k_sparsification: bool = True,
            use_normalization: bool = True
    ):
        """
        Create a dedicated optimizer used for `QSparseLocal`_ algorithm.

        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate.
            k: How many tensor components are kept during sparsification
            schedule: Number of rounds per synchronization round (Gap - 1)
            quantization_scheme: Sign or QSGD quantization
            sparsify: Whether sparsification takes place
            use_memory: Usage of error compensation
            quantization_levels: Number of quantization levels for QSGD quantization
            top_k_sparsification: Whether Topk or Randk sparsification is used
            use_normalization: Whether QSGD quantization uses normalization
        """

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0 < k:
            raise ValueError("Invalid k: {}".format(k))
        if not 0 < schedule:
            raise ValueError("Invalid schedule: {}".format(schedule))
        if not quantization_scheme == "sign" or quantization_scheme == "qsgd":
            raise ValueError("Invalid quanization scheme: {}".format(schedule))
        if not 0 < quantization_levels:
            raise ValueError("Invalid quantization level: {}".format(quantization_levels))
        
        defaults = dict(lr=lr, k=k, schedule=schedule)
        super(QSparseLocalOptimizer, self).__init__(params, defaults)
        self.step_id = 0
        self.schedule = schedule
        self.k = k
        self.lr = lr
        self.quantization_scheme = quantization_scheme
        self.sparsify = sparsify
        self.use_memory = use_memory
        self.quantization_levels = quantization_levels
        self.top_k_sparsification = top_k_sparsification
        self.use_normalization = use_normalization
        self.sync = True

        # initialize global and local model, and memory for error compensation
        for group_id, group in enumerate(self.param_groups):
            params_with_grad = []
            for p in group["params"]:
                params_with_grad.append(p)
                state = self.state[p]
                if len(state) == 0:
                    
                    state["global"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Set global to initialized value of local weights
                    state["global"].add_(p.data)
                    
                    state["memory"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["qsl_grad"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    """
                    Local parameter vector inside p.data
                    global: Global paramenter vector (same for all workers)
                    error_comp: Term for error compensation for quantization and sparsification of gradient tensor
                    qsl_grad: The quantized and sparsified gradient tensor to be sent to master (to allreduce)
                    """

    def __setstate__(self, state):
        super(QSparseLocalOptimizer, self).__setstate__(state)

    def step(self, closure=None):

        
        for group_id, group in enumerate(self.param_groups):
            for param_id, param in enumerate(group["params"]):
                state = self.state[param]

                # Calculate new global and local weights after synchronzation
                if self.sync:
                    state["global"].add_(state["qsl_grad"])
                    param.data.mul_(0).add_(state["global"])
                # Local update
                else:
                    param.data.add_(param.grad,alpha=-self.lr)
                    
        self.step_id += 1
        
        # Schedule defines the number of rounds per synchronization round
        self.sync = self.step_id % self.schedule == 0        

            
                
class QSparseLocalAlgorithmImpl(AlgorithmImpl):
    def __init__(
            self,
            process_group: BaguaProcessGroup,
            q_sparse_local_optimizer: QSparseLocalOptimizer,
            hierarchical: bool = True,
    ):
        super(QSparseLocalAlgorithmImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.optimizer = q_sparse_local_optimizer
        self.sync = self.optimizer.sync

    def need_reset(self):
        return False

    def init_tensors(self, bagua_distributed_data_parallel: BaguaDistributedDataParallel):
        parameters = bagua_distributed_data_parallel.bagua_build_params()

        for idx, (name, param) in enumerate(parameters.__reversed__()):
            param._q_sparse_local_name = name
            param._q_sparse_local_idx = idx

        tensor_groups = []
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                def set_weights(param, t):
                    self.optimizer.state[param]["qsl_grad"] = t

                registered_tensor = param.bagua_ensure_grad().ensure_bagua_tensor(
                    param._q_sparse_local_name,
                    bagua_distributed_data_parallel.bagua_module_name,
                    getter_closure=lambda param: self.optimizer.state[param]["qsl_grad"],
                    setter_closure=set_weights,
                )
                tensor_groups.append(registered_tensor)


        tensor_groups.sort(key=lambda x: x._q_sparse_local_idx)
        return tensor_groups

    def tensors_to_buckets(
            self, tensors: List[List[BaguaTensor]], do_flatten: bool
    ) -> List[BaguaBucket]:
        bagua_buckets = []
        for idx, bucket in enumerate(tensors):
            bagua_bucket = BaguaBucket(
                bucket,
                flatten=do_flatten,
                name=str(idx),
                alignment=self.process_group.get_global_communicator().nranks(),
            )
            bagua_buckets.append(bagua_bucket)
        return bagua_buckets

    def init_operations(
            self,
            bagua_distributed_data_parallel: BaguaDistributedDataParallel,
            bucket: BaguaBucket,
    ):
        bucket.clear_ops()      
        def preprocess(*args):
            for group_id, group in enumerate(self.optimizer.param_groups):
                for param_id, param in enumerate(group["params"]):
                    state = self.optimizer.state[param]
                     
                    if self.optimizer.sync:
                        ##### Before allreduce operation
                        # Compute temporary local parameter vector
                        # Compute compressed gradient and update error compensation term
                        
                        param.data.add_(param.grad,alpha=-self.optimizer.lr)

                        # No output, new values assigned inside the function
                        qsl(param.data - state["global"],
                            state["memory"],
                            state["qsl_grad"],
                            k = self.optimizer.k,
                            topK_flag=self.optimizer.top_k_sparsification,
                            s=self.optimizer.quantization_levels,
                            sparsify = self.optimizer.sparsify,
                            use_memory = self.optimizer.use_memory,
                            quantization_scheme = self.optimizer.quantization_scheme,
                            use_normalization = self.optimizer.use_normalization
                        )


        bucket.append_python_op(lambda *args :preprocess(*args), group=self.process_group)

        bucket.append_centralized_synchronous_op(
            hierarchical=self.hierarchical,
            average=True,
            scattergather=True,
            compression="MinMaxUInt8",    
            group=self.process_group,
        )

    # We use a qsl_gradient hook
    def init_backward_hook(self, bagua_distributed_data_parallel: BaguaDistributedDataParallel):
        
        def hook_qsl_grad(parameter_name, parameter):
            assert (
                    parameter.bagua_backend_tensor().data_ptr()
                    == self.optimizer.state[parameter]["qsl_grad"].data_ptr()
            ), "bagua backend tensor data_ptr should match _q_sparse_local_grad data_ptr"

            if self.optimizer.sync:
                parameter.bagua_mark_communication_ready()

        return hook_qsl_grad


class QSparseLocalAlgorithm(Algorithm):
    def __init__(self, q_sparse_local_optimizer: QSparseLocalOptimizer, hierarchical: bool = True):
        self.hierarchical = hierarchical
        self.optimizer = q_sparse_local_optimizer

    def reify(self, process_group: BaguaProcessGroup) -> QSparseLocalAlgorithmImpl:
        return QSparseLocalAlgorithmImpl(
            process_group,
            q_sparse_local_optimizer=self.optimizer,
            hierarchical=self.hierarchical,
        )
