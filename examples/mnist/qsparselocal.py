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

sparsify = True
use_memory = True
quantization_scheme = 'sign'
quantization_levels = 256
top_k_sparsification = True
k = 1000
use_normalization = True


## input: Uncompressed Gradient tensor
## Output: Quantized and sparsified Gradient tensor
def qsl(eta_grad,
        memory,
        qsl_grad,   # No output, instead change in function
        topK_flag,
        s,
        # sparsify,
        # use_memory,
        # quantization_scheme,
        # use_normalization
        ):
    ###To do: Allow other quantization
    def signq(var):
        # Normalization according to input
        # ||var||_1 * sign(var)/
        one_norm = torch.norm(var, p=1)
        return one_norm * torch.sign(var + 1e-13) / float(torch.numel(var))
        # return torch.sign(var)  # Returns a new tensor with the signs of the elements of input

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

        return q, input - q

    input = memory + eta_grad

    org_shape = input.size()
    numel = torch.numel(input)
    K = min(numel, k)  # k is the optimizer's k,
    # K is the actual value used for sparsification

    if topK_flag:
        # Get values and index tensor of chosen components
        # flat shape with absolute values
        _, indices = torch.topk(torch.reshape(torch.abs(input), [-1]), K)
    else:
        indices = torch.from_numpy(np.random.choice(torch.range(numel), K, False))

    # Flatten input
    flat_input = torch.reshape(input, [-1])
    values = torch.gather(flat_input, 0, indices)  # dim=0
    norm1 = torch.norm(values)
    quantization_func = get_quantization(quantization_scheme)
    flattened_quantized = torch.zeros_like(flat_input).scatter(0, indices,
                                                               quantization_func(values))
    quantization = torch.reshape(flattened_quantized, shape=org_shape)

    q_func = lambda: quantization
    zero_tensor = lambda: torch.zeros_like(input, dtype=torch.float32)

    # q = torch.where( float(0)<norm1, q_func, zero_tensor)    # Where not applicable for choosing functions
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
            lr: float = 1e-3,  ## Later step dependent learning rate
            k: int = 1000,
            schedule: int = 1
    ):
        """
        Create a dedicated optimizer used for `QSparseLocal`_ algorithm.

        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate.
            k: How many tensor components are kept during sparsification
            schedule: Number of rounds per synchronization round (Gap - 1)
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0 < schedule:
            raise ValueError("Invalid schedule: {}".format(lr))

        defaults = dict(lr=lr, k=k, schedule=schedule)
        super(QSparseLocalOptimizer, self).__init__(params, defaults)
        self.step_id = 0
        self.schedule = schedule
        self.k = k
        self.lr = lr
        self.sync = False

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

                #Calculate new global and local weights after synchronzation
                if self.sync:
                    state["global"].add_(state["qsl_grad"])
                    param.data.mul_(0).add_(state["global"])
                    
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
        """
        Implementation of the `QSparseLocal Algorithm `.

        Args:
            process_group: The process group to work on.
            q_sparse_local_optimizer: A QSparseLocalOptimizer initialized with model parameters.
            hierarchical: Enable hierarchical communication.
        """
        super(QSparseLocalAlgorithmImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.optimizer = q_sparse_local_optimizer
        self.sync = self.optimizer.sync

    def need_reset(self):
        if self.optimizer.sync or \
          self.optimizer.step_id%self.optimizer.schedule == self.optimizer.schedule-1:
            return True
        else:
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
                    # Set compressed gradient to mean of all workers' compressed gradients
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
        # For synchronization round we utilize allreduce, else no synchronization takes place
        if self.optimizer.sync:
            def preprocess(*args):
              for group_id, group in enumerate(self.optimizer.param_groups):
                for param_id, param in enumerate(group["params"]):
                    state = self.optimizer.state[param]
                    ##### Before allreduce operation
                    # Compute temporary local parameter vector
                    # Compute compressed gradient and new error compensation term
                    param.data.add_(param.grad,alpha=-self.optimizer.lr) 
                    if self.optimizer.sync:
                        # No output, new values assigned inside the function
                        qsl(param.data - state["global"], state["memory"],state["qsl_grad"],
                                                    topK_flag=top_k_sparsification, s=quantization_levels)

            bucket.append_python_op(lambda *args :preprocess(*args), group=self.process_group)

            # Compression is done by Bagua

            bucket.append_centralized_synchronous_op(
                hierarchical=self.hierarchical,
                average=True,  # Maybe try false and then average it manually to preserve ints
                scattergather=True,
                compression="MinMaxUInt8",    
                group=self.process_group,
            )
        else:  # Nothing happens
            pass

    # Instead of momentum hook, we use a qsl_gradient hook
    def init_backward_hook(self, bagua_distributed_data_parallel: BaguaDistributedDataParallel):

        def hook_qsl_grad(parameter_name, parameter):
            assert (
                    parameter.bagua_backend_tensor().data_ptr()
                    == self.optimizer.state[parameter]["qsl_grad"].data_ptr()
            ), "bagua backend tensor data_ptr should match _q_sparse_local_grad data_ptr"
            parameter.bagua_mark_communication_ready()

        return hook_qsl_grad


class QSparseLocalAlgorithm(Algorithm):
    def __init__(self, q_sparse_local_optimizer: QSparseLocalOptimizer, hierarchical: bool = True):
        """
        Create an instance of the `QSparseLocal Algorithm' .

        Args:
            q_sparse_local_optimizer: A QSparseLocalOptimizer initialized with model parameters.
            hierarchical: Enable hierarchical communication.
        """
        self.hierarchical = hierarchical
        self.optimizer = q_sparse_local_optimizer

    def reify(self, process_group: BaguaProcessGroup) -> QSparseLocalAlgorithmImpl:
        return QSparseLocalAlgorithmImpl(
            process_group,
            q_sparse_local_optimizer=self.optimizer,
            hierarchical=self.hierarchical,
        )
