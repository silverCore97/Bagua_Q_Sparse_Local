#!/usr/bin/env python3
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup
from torch.optim.optimizer import Optimizer
import torch
import math
from typing import List, Tuple


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
                    #state["local"] = torch.zeros_like(
                    #     p, memory_format=torch.preserve_format
                    #)

                    state["global"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["error_comp"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["qsl_grad"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    """
                    ###local: Local parameter vector
                    global: Global paramenter vector (same for all workers)
                    error_comp: Term for error compensation for quantization and sparsification of gradient tensor
                    qsl_grad: The quantized and sparsified gradient tensor to be sent to master (to allreduce)
                    """
                    # Set local weights to 0 in the beginning, format: torch.Size([32, 1, 3, 3])
                    p.data = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

    def __setstate__(self, state):
        super(QSparseLocalOptimizer, self).__setstate__(state)

    def step(self, closure=None):
        self.step_id += 1
        for group_id, group in enumerate(self.param_groups):


            for param_id, param in enumerate(group["params"]):
                state = self.state[param]


                # Local weights equal to local model
                # Step 5
                # For step_id==1 this part is not supposed to do anything
                #print("Gradient Size:", param.grad.size())
                #print("Gradient Size Position 0:", param.grad[0])
                state["global"].add_(state["qsl_grad"],alpha=-1)
                param.data = state["global"]

                ##### Step 2
                # Compute compressed gradient using the gradient from backwards propagation

                ## input: Uncompressed Gradient tensor
                ## Output: Quantized and sparsified Gradient tensor
                def qsl(self, input):
                    ###To do: Allow other quantization
                    def signq(self, var):
                        return torch.sign(
                            var)  # Returns a new tensor with the signs of the elements of input

                    ### To do: Allow rand_k sparsification
                    org_shape = input.size()
                    numel = torch.numel(input)
                    K = min(numel, self.k)  # k is the optimizer's k,
                    # K is the actual value used for sparsification

                    # Flatten input
                    flat_input = torch.reshape(input, [-1])

                    # Get values and index tensor of chosen components (Set dim=-1 for testing)
                    values, indices = torch.topk(flat_input, K,
                                                 dim=-1)  # flat_input instead of input

                    # torch.zeros() on cpu not cuda, use torch.zeros_like() with memory_format=torch.preserve_format as default
                    # flat_quantized = torch.zeros(flat_input.size()).scatter(0, indices, signq(self,values))

                    flat_quantized = torch.zeros_like(flat_input).scatter(0, indices,
                                                                          signq(self, values))
                    quantized = torch.reshape(flat_quantized, shape=org_shape)

                    return quantized

                # Update compressed gradient
                state["qsl_grad"] = \
                    qsl(self, state["error_comp"].add_(state["global"]).add_(param.grad, alpha=-self.lr))

                # Update error_comp
                state["error_comp"].add_(state["qsl_grad"], alpha=-1)


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

            ##### Step 3
            bucket.append_centralized_synchronous_op(
                hierarchical=self.hierarchical,
                average=True,
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
