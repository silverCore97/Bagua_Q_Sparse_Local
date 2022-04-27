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
        is_next_level = (torch.random_uniform(shape=var.size(), dtype=torch.float32) <
                         (level_float - previous_level))
        is_next_level = float(is_next_level)
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

    #print("\n\n input in qsl","of shape",input.size(),"\n", torch.reshape(input, [-1])[:3])
    #print("q:",torch.reshape(q, [-1])[:3])
    #print("err:", torch.reshape(err, [-1])[:3])

    return q, err


class QSparseLocalOptimizer(Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-3,  ## Later step dependent learning rate
            k: int = 1000,
            schedule: List = None
    ):
        """
        Create a dedicated optimizer used for
        `QSparseLocal <https://tutorials.baguasys.com/algorithms/q-adam>`_ algorithm.

        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate.
            k: How many tensor components are kept during sparsification
            schedule: Description of synchronization schedule
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, k=k, schedule=schedule)
        super(QSparseLocalOptimizer, self).__init__(params, defaults)
        # TODO: QSparseLocal optimizer maintain `step_id` in its state
        self.step_id = 0
        self.schedule = schedule
        self.k = k
        self.lr = lr

        # initialize global and local model, and memory for error compensation
        for group_id, group in enumerate(self.param_groups):
            params_with_grad = []
            for p in group["params"]:
                params_with_grad.append(p)
                state = self.state[p]
                if len(state) == 0:
                    ##### Instead of state["local"] use param.data as local model
                    # state["local"] = torch.zeros_like(
                    #     p, memory_format=torch.preserve_format
                    # )

                    state["global"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Set global to initialized value of local weights
                    state["global"].add_(p.data)

                    # print("global init",state["global"].size())
                    state["memory"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # print("comp init", state["error_comp"].size())
                    state["qsl_grad"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    #print("Hello")
                    """
                    ###local: Local parameter vector
                    global: Global paramenter vector (same for all workers)
                    error_comp: Term for error compensation for quantization and sparsification of gradient tensor
                    qsl_grad: The quantized and sparsified gradient tensor to be sent to master (to allreduce)
                    """
                    """  ###Leads to problem with gradient
                    # Set local weights to 0 in the beginning, format: torch.Size([32, 1, 3, 3])
                    
                    # p.data uses normal pytorch initializer by default
                    p.data = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    """

    def __setstate__(self, state):
        super(QSparseLocalOptimizer, self).__setstate__(state)

    def step(self, closure=None):
        self.step_id += 1

        for group_id, group in enumerate(self.param_groups):

            for param_id, param in enumerate(group["params"]):
                state = self.state[param]

                #print("Data: ",torch.reshape(param.data,[-1])[:10])

                #Calculate allreduce
                if self.schedule is None or self.optimizer.step_id in self.schedule:
                    #### In train resnet
                    #new_var = last_var + hvd.allreduce(var-last_var, var, opt, wipe_memory)
                    state["global"].add_(state["qsl_grad"])
                    param.data.add_(-param.data).add_(state["global"])


                #### In train resnet
                # grad was already applied
                #print("Gradient:",param.grad)   #Exists

                # Converges with only this
                param.data.add_(param.grad, alpha=-self.lr)

                ##### In allreduce before allreduce operation
                # Compute compressed gradient using the gradient from backwards propagation
                # eta local-global

                if self.schedule is None or self.optimizer.step_id in self.schedule:
                    # print("Data::", param.data)
                    # print("global",state["global"])
                    #print("qsl:", state["qsl_grad"])
                    # print("memory", state["memory"])
                    state["qsl_grad"], state["memory"] = qsl(param.data - state["global"], state["memory"],
                                                             topK_flag=top_k_sparsification, s=quantization_levels)
                    #print("qsl2:", state["qsl_grad"])
                    # print("memory2", state["memory"])


class QSparseLocalAlgorithmImpl(AlgorithmImpl):
    def __init__(
            self,
            process_group: BaguaProcessGroup,
            q_sparse_local_optimizer: QSparseLocalOptimizer,
            hierarchical: bool = True,
    ):
        """
        Implementation of the
        `QSparseLocal Algorithm <https://tutorials.baguasys.com/algorithms/q-adam>`_
        .
        Args:
            process_group: The process group to work on.
            q_sparse_local_optimizer: A QSparseLocalOptimizer initialized with model parameters.
            hierarchical: Enable hierarchical communication.
        """
        super(QSparseLocalAlgorithmImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.optimizer = q_sparse_local_optimizer
        self.schedule = self.optimizer.schedule

    # def need_reset(self):
    #    return True

    def init_tensors(self, bagua_distributed_data_parallel: BaguaDistributedDataParallel):
        parameters = bagua_distributed_data_parallel.bagua_build_params()

        for idx, (name, param) in enumerate(parameters.__reversed__()):
            param._q_sparse_local_name = name
            param._q_sparse_local_idx = idx

        tensor_groups = []
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if self.schedule is None or self.optimizer.step_id in self.schedule:

                    # Second half Step 4
                    def set_weights(param, t):
                        # Set compressed gradient to mean of all workers compressed gradients
                        self.optimizer.state[param]["qsl_grad"] = t

                    registered_tensor = param.bagua_ensure_grad().ensure_bagua_tensor(
                        param._q_sparse_local_name,
                        bagua_distributed_data_parallel.bagua_module_name,
                        getter_closure=lambda param: self.optimizer.state[param]["qsl_grad"],
                        setter_closure=set_weights,
                    )
                    tensor_groups.append(registered_tensor)
                else:
                    # Nothing happens
                    pass

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
                # flatten=True,
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
        if self.schedule is None or self.optimizer.step_id in self.schedule:

            # Compression is done by Bagua
            bucket.append_centralized_synchronous_op(
                hierarchical=self.hierarchical,
                average=True,  # Maybe try false and then average it manually to preserve ints
                scattergather=True,
                compression="MinMaxUInt8",      #TO DO:  Make qsl_grad suitable for compression
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
        Create an instance of the
        `QSparseLocal Algorithm <https://tutorials.baguasys.com/algorithms/q-adam>`_
        .

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
