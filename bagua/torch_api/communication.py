import logging
import time
import io
import pickle
import multiprocessing
import bagua_core as B
from bagua.service import AutotuneService
from . import env
from .env import (
    get_master_addr,
    get_world_size,
    get_rank,
    get_local_rank,
    get_node_rank,
    get_default_bucket_size,
    get_bagua_service_port,
    get_autotune_server_wait_time,
    find_free_network_port,
)
from enum import IntEnum
from .utils import flatten, unflatten
import torch
import torch.distributed as dist
from bagua.service.autotune_service import AutotuneClient
from functools import lru_cache
from datetime import timedelta
from typing import Optional, List
import torch.distributed.distributed_c10d as c10d
from torch.distributed import ProcessGroup as TorchProcessGroup
import gorilla
import weakref

# fmt: off
__all__ = [
    "ReduceOp", "new_group", "from_torch_group", "init_process_group",
    "is_initialized", "send", "recv", "broadcast", "broadcast_object",
    "reduce", "reduce_inplace", "allreduce", "allreduce_inplace",
    "allgather", "allgather_inplace", "gather", "gather_inplace",
    "scatter", "scatter_inplace", "reduce_scatter", "reduce_scatter_inplace",
    "alltoall", "alltoall_inplace", "barrier", "BaguaProcessGroup"
]

# Process group's global rank to local rank mapping
_pg_group_ranks = {}

# Process group's name to BaguaProcessGroup
_pg_map = {}

# Default process group state
_default_pg = None

# Default store
_default_store = None

# Process group count for default naming
_group_count = 0

# Torch process group to bagua process group
_torch_to_bagua_pg_map = weakref.WeakKeyDictionary({})


# must be consistent with Aluminum ReductionOperator: https://github.com/BaguaSys/Aluminum/blob/master/include/aluminum/base.hpp
class ReduceOp(IntEnum):
    """An enum-like class for available reduction operations: ``SUM``, ``PRODUCT``, ``MIN``, ``MAX``, ``BAND``,
    ``BOR``, ``BXOR`` and ``AVG``."""

    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3
    BOR = 7
    BAND = 8
    BXOR = 9
    AVG = 10


@gorilla.patches(TorchProcessGroup, filter=lambda name, obj: "bagua" in name)
class BaguaProcessGroupPatch:
    def bagua_patch(self, stream: Optional[torch.cuda.Stream] = None):
        global _torch_to_bagua_pg_map
        if self not in _torch_to_bagua_pg_map:
            _torch_to_bagua_pg_map[self] = from_torch_group(self, stream)

        return self

    @property
    def bagua_pg(self):
        assert self in _torch_to_bagua_pg_map, "cannot find associated Bagua process group in cache, BaguaProcessGroupPatch.bagua_patch(...) needs to be run first to initialize Bagua process group in cache."
        return _torch_to_bagua_pg_map[self]

    def bagua_get_global_communicator(self):
        return get_communicator(self.bagua_pg.group_name, "global")

    def bagua_get_inter_node_communicator(self):
        return get_communicator(self.bagua_pg.group_name, "inter")

    def bagua_get_intra_node_communicator(self):
        return get_communicator(self.bagua_pg.group_name, "intra")


_base = gorilla._get_base(BaguaProcessGroupPatch)
_decorator_data = gorilla.get_decorator_data(_base)
for patch in _decorator_data.patches:
    gorilla.apply(patch)


class BaguaProcessGroup:
    """Definition of Bagua process group."""
    def __init__(self, ranks, stream, group_name):
        self.ranks = ranks
        self.stream = stream
        self.group_name = group_name
        logging.debug(f"Initialize Bagua process group of ranks {self.ranks}")

    def _get_intra_ranks(self):
        rank_mappings = _get_rank_mappings()

        intra_ranks = list(
            filter(
                lambda rank: rank_mappings[rank][0] == get_node_rank(),
                self.ranks,
            )
        )
        return intra_ranks

    def _get_inter_ranks(self):
        rank_mappings = _get_rank_mappings()

        inter_ranks = list(
            filter(
                lambda rank: rank_mappings[rank][1] == rank_mappings[self.ranks[0]][1],
                self.ranks,
            )
        )
        return inter_ranks

    def get_global_communicator(self) -> B.BaguaSingleCommunicatorPy:
        """Returns the global communicator of current process group."""
        return get_communicator(self.group_name, "global")

    def get_inter_node_communicator(self) -> B.BaguaSingleCommunicatorPy:
        """Returns the inter-node communicator of current process group."""
        return get_communicator(self.group_name, "inter")

    def get_intra_node_communicator(self) -> B.BaguaSingleCommunicatorPy:
        """Returns the intra-node communicator of current process group."""
        return get_communicator(self.group_name, "intra")


@lru_cache(maxsize=None)
def _get_rank_mappings():
    rank_mappings = {}

    rank_tensors = torch.cuda.LongTensor(get_world_size(), 2)
    rank_tensors[get_rank()][0] = get_node_rank()
    rank_tensors[get_rank()][1] = get_local_rank()
    allgather_inplace(rank_tensors)

    for i in range(get_world_size()):
        rank_mappings[i] = rank_tensors[i][0].item(), rank_tensors[i][1].item()

    return rank_mappings


def _check_default_pg():
    """
    Helper that checks if the default process group has been initialized, with
    assertion.

    """
    assert _default_pg is not None, "Default process group is not initialized"


def is_initialized():
    """
    Checking if the default process group has been initialized.

    """
    return _default_pg is not None


def _get_default_group():
    """
    Getting the default process group created by :func:`init_process_group`.

    """
    if not is_initialized():
        raise RuntimeError(
            "Default process group has not been initialized, "
            "please make sure to call init_process_group."
        )
    return _default_pg


def _bagua_backend_comm(comm: Optional[B.BaguaSingleCommunicatorPy] = None):
    """
    Return ``None`` if the current process's rank is not in a given communicator.
    Otherwise return the communicator passed in.
    """
    if _rank_not_in_comm(comm):
        return None
    return comm


def new_group(
    ranks: Optional[List[int]] = None, stream: Optional[torch.cuda.Stream] = None
) -> BaguaProcessGroup:
    """
    Creates a new process group.

    This function requires that all processes in the default group (i.e. all
    processes that are part of the distributed job) enter this function, even
    if they are not going to be members of the group. Additionally, groups
    should be created in the same order in all processes.

    Each process group will create three communicators on request, a global communicator,
    a inter-node communicator and a intra-node communicator. Users can access them through
    ``group.get_global_communicator()``, ``group.get_inter_node_communicator()``
    and ``group.get_intra_node_communicator()`` respectively.

    Args:
        ranks: List of ranks of group members. If ``None``, will be
            set to all ranks. Default is ``None``.
        stream: A CUDA stream used to execute NCCL operations. If ``None``,
            CUDA stream of the default group will be used. See
            `CUDA semantics <https://pytorch.org/docs/stable/notes/cuda.html?highlight=stream>`_
            for details.

    Returns:
        A handle of process group that can be given to collective calls.

    .. note::
        The global communicator is used for global communications involving all ranks in the process group.
        The inter-node communicator and the intra-node communicator is used for hierarchical communications
        in this process group.

    .. note::
        For a specific communicator ``comm``, ``comm.rank()`` returns the rank of current process and
        ``comm.nranks()`` returns the size of the communicator.
    """
    global _group_count
    global _pg_group_ranks
    global _pg_map

    _group_count += 1

    if ranks is None:
        ranks = list(range(get_world_size()))
    else:
        # sanity check for the input ranks
        for rank in ranks:
            if rank < 0 or rank >= get_world_size():
                raise ValueError(
                    "Invalid rank {}, should be non-negative and less than world size {}.",
                    rank,
                    get_world_size(),
                )
        ranks = sorted(ranks)

    if stream is None:
        _check_default_pg()
        stream = _get_default_group().stream

    group_name = str(_group_count)
    pg = BaguaProcessGroup(ranks, stream, str(_group_count))
    # Create the global rank to group rank mapping
    _pg_group_ranks[pg] = {
        global_rank: group_rank for group_rank, global_rank in enumerate(ranks)
    }
    _pg_map[group_name] = pg

    return pg


__torch_group_id = 0


def from_torch_group(group, stream: Optional[torch.cuda.Stream] = None) -> BaguaProcessGroup:
    """
    Convert a Pytorch process group to its equivalent Bagua process group.

    Args:
        group: A handle of the Pytorch process group.
        stream: A CUDA stream used to execute NCCL operations. If ``None``,
            CUDA stream of the default group will be used. See :func:`new_group`
            for more information.

    Returns:
       A handle of the Bagua process group.
    """
    global __torch_group_id

    torch_group_id = __torch_group_id
    __torch_group_id += 1

    ranks = None
    if group in c10d._pg_group_ranks:
        ranks = list(c10d._pg_group_ranks[group].keys())
    elif _default_store:
        def rank_key(rank):
            return "global rank of {}.{}".format(torch_group_id, rank)

        _default_store.set(rank_key(group.rank()), env.get_rank())
        ranks = [int(_default_store.get(rank_key(i))) for i in range(group.size())]
    else:
        ranks = list(range(group.size()))

    return new_group(ranks, stream)


@lru_cache(maxsize=None)
def get_communicator(group_name: str, comm_name: str):
    global _pg_map

    pg = _pg_map[group_name]
    if comm_name == "global":
        ranks = pg.ranks
    elif comm_name == "inter":
        ranks = pg._get_inter_ranks()
    elif comm_name == "intra":
        ranks = pg._get_intra_ranks()
    else:
        raise ValueError("comm_name should be one of ['global', 'inter', 'intra']")

    comm_key = "{}_{}_{}".format(group_name, comm_name, ",".join(map(str, ranks)))

    nccl_unique_id = broadcast_nccl_unique_id(comm_key, root=ranks[0])

    if get_rank() not in ranks:
        return CommMember.NON_COMM_MEMBER

    rank = ranks.index(get_rank())
    nranks = len(ranks)

    comm = B.BaguaSingleCommunicatorPy(
        rank=rank,
        nranks=nranks,
        device_id=get_local_rank(),
        stream_ptr=pg.stream.cuda_stream,
        nccl_unique_id_str=nccl_unique_id,
    )

    logging.debug(
        "init bagua communicator %s-%s ok, global rank: %s rank: %s",
        group_name,
        comm_name,
        get_rank(),
        comm.rank(),
    )
    comm.cuda_stream = pg.stream
    return comm


def _rank_not_in_comm(comm: Optional[B.BaguaSingleCommunicatorPy] = None):
    """
    Return ``True`` if the current process's rank is not in a given communicator.

    """
    if comm is None:
        return False
    return comm == CommMember.NON_COMM_MEMBER


def _rank_not_in_group(group: Optional[BaguaProcessGroup] = None):
    """
    Return ``True`` if the current process is not in a given process group.

    """
    if group is None:
        return False
    return _rank_not_in_comm(group.get_global_communicator())




@lru_cache(maxsize=None)
def get_backend(model_name: str):
    backend = B.BaguaCommBackendPy(100, device_id=get_local_rank())
    backend.model_name = model_name
    return backend


def run_flask_app(port):
    from flask import Flask
    from gevent.pywsgi import WSGIServer
    import os

    os.environ["WERKZEUG_RUN_MAIN"] = "true"

    autotune_service = AutotuneService(
        world_size=get_world_size(),
        autotune_level=env.get_autotune_level(),
        max_samples=env.get_autotune_max_samples(),
        sampling_confidence_time_s=env.get_autotune_sampling_confidence_time_s(),
        warmup_time_s=env.get_autotune_warmup_time_s(),
        is_output_autotune_log=env.get_is_output_autotune_log(),
        default_bucket_size=get_default_bucket_size(),
    )
    app = Flask(__name__)
    app = autotune_service.setup_app(app)

    http_server = WSGIServer(
        listener=("0.0.0.0", port),
        application=app,
        log=None,
    )
    http_server.serve_forever()


_autotune_server = None
_autotune_service_port = None


def start_autotune_server(service_port: int):
    """Starts autotune server in background."""
    global _autotune_server

    _autotune_server = multiprocessing.Process(target=run_flask_app, args=(service_port, ))
    _autotune_server.daemon = True
    _autotune_server.start()


@lru_cache(maxsize=None)
def get_hyperparameters_service_client():
    global _autotune_service_port
    hyperparameters_service_client = AutotuneClient(
        get_master_addr(), _autotune_service_port
    )
    return hyperparameters_service_client


def _find_free_bagua_service_port(store) -> int:
    service_port = get_bagua_service_port()
    if service_port > 0:
        return service_port

    if get_rank() == 0:
        service_port = find_free_network_port()
        store.set("bagua_service_port", str(service_port))
    else:
        service_port = int(store.get("bagua_service_port"))

    return service_port


def init_process_group(store: Optional[torch.distributed.Store] = None):
    """Initializes the PyTorch builtin distributed process group, and this will
    also initialize the distributed package, should be executed before all the
    APIs of Bagua.

    Args:
        store: Key/value store accessible to all workers, used to exchange
            connection/address information. If ``None``, a TCP-based store will be created.
            Default: ``None``.

    Examples::
        >>> import torch
        >>> import bagua.torch_api as bagua
        >>>
        >>> torch.cuda.set_device(bagua.get_local_rank()) # THIS LINE IS IMPORTANT. See the notes below.
        >>> bagua.init_process_group()
        >>>
        >>> model = torch.nn.Sequential(
        ...    torch.nn.Linear(D_in, H),
        ...    torch.nn.ReLU(),
        ...    torch.nn.Linear(H, D_out),
        ...    )
        >>> optimizer = torch.optim.SGD(
        ...    model.parameters(),
        ...    lr=0.01,
        ...    momentum=0.9
        ...    )
        >>> model = model.with_bagua([optimizer], ...)

    .. note::
        Each process should be associated to a CUDA device using `torch.cuda.set_device()`,
        before calling :meth:`init_process_group`. Otherwise you may encounter the
        `fatal runtime error: Rust cannot catch foreign exceptions` error.
    """


    global _default_pg
    global _default_store
    global _autotune_service_port

    if _default_pg is not None:
        raise RuntimeError("trying to initialize the default process group twice!")

    if _default_store is not None:
        raise RuntimeError("The default store has been initialized else where!")

    if store is None:
        timeout = timedelta(minutes=30)
        store, _, _ = next(torch.distributed.rendezvous(url="env://", timeout=timeout))
        store.set_timeout(timeout)
        _default_store = store
    else:
        _default_store = store

    _autotune_service_port = _find_free_bagua_service_port(_default_store)
    if get_rank() == 0 and _autotune_server is None:
        start_autotune_server(_autotune_service_port)

    AUTOTUNE_SERVER_WAIT_TIME = 30
    wait_time = get_autotune_server_wait_time()
    # at least wait 30 seconds
    if wait_time < AUTOTUNE_SERVER_WAIT_TIME:
        wait_time = AUTOTUNE_SERVER_WAIT_TIME

    start = time.time()
    service_ready = False
    while (time.time() - start) < wait_time:
        client = get_hyperparameters_service_client()
        service_ready = client.health_check()
        if service_ready:
            break
    if not service_ready:
        raise Exception("Warning! autotune service not ready after {} seconds. "
                        "You can adjust this duration through "
                        "`BAGUA_AUTOTUNE_SERVER_WAIT_TIME` environment variable.".format(wait_time))

    # TODO remove the dependency on torch process group
    if not dist.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",
            store=_default_store,
            rank=get_rank(),
            world_size=get_world_size(),
        )  # fmt: off

    _default_pg = new_group(stream=torch.cuda.Stream(priority=-1))


def broadcast_nccl_unique_id(comm_key: str, root):
    global _default_store
    if get_rank() == root:
        idstr = B.BaguaSingleCommunicatorPy.generate_nccl_unique_id_str()
        _default_store.set(comm_key, idstr)
    else:
        idstr = _default_store.get(comm_key)
        idstr = str(idstr, encoding="utf-8")

    return idstr


class comm(object):
    WORLD = object()


class CommMember(object):
    # Alias to group.WORLD for backward compatibility
    WORLD = comm.WORLD
    NON_COMM_MEMBER = object()


def send(tensor: torch.Tensor, dst: int, comm: Optional[B.BaguaSingleCommunicatorPy] = None):
    r"""Sends a tensor to :attr:`dst` synchronously.

    Args:
        tensor: Tensor to send.
        dst: Destination rank.
        comm: A handle of the Bagua communicator to work on. By default, the global
             communicator of the default process group will be used.
    """
    if _rank_not_in_comm(comm):
        return

    assert tensor.device != torch.device("cpu"), "input tensor must be CUDA and dense"

    if comm is None or comm is CommMember.WORLD:
        comm = _get_default_group().get_global_communicator()

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        comm.send(tensor.to_bagua_tensor().bagua_backend_tensor(), dst)

    comm.cuda_stream.synchronize()


def recv(tensor: torch.Tensor, src: int, comm: Optional[B.BaguaSingleCommunicatorPy] = None):
    r"""Receives a tensor synchronously.

    Args:
        tensor: Tensor to fill with received data.
        src: Source rank.
        comm: A handle of the Bagua communicator to work on. By default, the global
             communicator of the default process group will be used.
    """
    if _rank_not_in_comm(comm):
        return

    assert tensor.device != torch.device("cpu"), "input tensor must be CUDA and dense"

    if comm is None or comm is CommMember.WORLD:
        comm = _get_default_group().get_global_communicator()

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        comm.recv(tensor.to_bagua_tensor().bagua_backend_tensor(), src)

    comm.cuda_stream.synchronize()


def broadcast_coalesced(tensors, src=0, comm: Optional[B.BaguaSingleCommunicatorPy] = None):

    if _rank_not_in_comm(comm):
        return

    for tensor in tensors:
        assert tensor.device != torch.device(
            "cpu"
        ), "input tensors must be CUDA and dense"

    if comm is None or comm is CommMember.WORLD:
        comm = _get_default_group().get_global_communicator()

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        coalesced = flatten(tensors)
        comm.broadcast(coalesced.to_bagua_tensor().bagua_backend_tensor(), src)
        for buf, synced in zip(tensors, unflatten(coalesced, tensors)):
            buf.copy_(synced)

    # TODO: remove
    comm.cuda_stream.synchronize()


# Copyright 2020 Uber Technologies, Inc. All Rights Reserved.
# Copyright (c) 2021 Kuaishou AI Platform & DS3 Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This function is copied fron Hovorod: https://github.com/horovod/horovod
# with minor changes.
def broadcast_object(obj: object, src: int = 0, comm: Optional[B.BaguaSingleCommunicatorPy] = None) -> object:
    """Serializes and broadcasts an object from root rank to all other processes.
    Typical usage is to broadcast the ``optimizer.state_dict()``, for example:

        >>> state_dict = broadcast_object(optimizer.state_dict(), 0)
        >>> if get_rank() > 0:
        >>>     optimizer.load_state_dict(state_dict)


    Args:
        obj: An object capable of being serialized without losing any context.
        src: The rank of the process from which parameters will be broadcasted to all other processes.
        comm: A handle of the Bagua communicator to work on. By default, the global
             communicator of the default process group will be used.
    Returns:
        The object that was broadcasted from the :attr:`src`.

    .. note::
        This operation will move data to GPU before communication and back to CPU after communication, and it requires
        CPU-GPU synchronization.
    """

    if get_rank() == src:
        b = io.BytesIO()
        pickle.dump(obj, b)
        t = torch.cuda.ByteTensor(bytearray(b.getvalue()))
        # TODO: use IntTensor after int32 communication is supported
        sz = torch.cuda.LongTensor([t.shape[0]])
        broadcast(sz, src, comm)
    else:
        sz = torch.cuda.LongTensor([0])
        broadcast(sz, src, comm)
        t = torch.cuda.ByteTensor(sz.tolist()[0])

    broadcast(t, src, comm)

    if get_rank() != src:
        buf = io.BytesIO(t.cpu().numpy().tobytes())
        obj = pickle.load(buf)

    return obj


def broadcast(tensor: torch.Tensor, src: int = 0, comm: Optional[B.BaguaSingleCommunicatorPy] = None):
    r"""Broadcasts the tensor to all processes associated with the communicator.

    :attr:`tensor` must have the same number of elements in all processes
    participating in the collective.

    Args:
        tensor: Data to be sent if :attr:`src` is the rank of
            current process, and tensor to be used to save received data
            otherwise.
        src: Source rank. Default: 0.
        comm: A handle of the Bagua communicator to work on. By default, the global
             communicator of the default process group will be used.
    """

    if _rank_not_in_comm(comm):
        return

    assert tensor.device != torch.device("cpu"), "input tensor must be CUDA and dense"

    if comm is None or comm is CommMember.WORLD:
        comm = _get_default_group().get_global_communicator()

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        comm.broadcast(tensor.to_bagua_tensor().bagua_backend_tensor(), src)

    # TODO: remove
    comm.cuda_stream.synchronize()


def reduce(
    send_tensor: torch.Tensor,
    recv_tensor: torch.Tensor,
    dst: int,
    op: ReduceOp = ReduceOp.SUM,
    comm: Optional[B.BaguaSingleCommunicatorPy] = None,
):
    r"""Reduces the tensor data across all processes.

    Only the process whit rank :attr:`dst` is going to receive the final result.

    Args:
        send_tensor: Input of the collective.
        recv_tensor: Output of the collective, must have the same size with :attr:`send_tensor`.
        dst: Destination rank.
        op: One of the values from :class:`ReduceOp`
            enum. Specifies an operation used for element-wise reductions.
        comm: A handle of the Bagua communicator to work on. By default, the global
             communicator of the default process group will be used.
    """

    if _rank_not_in_comm(comm):
        return

    assert send_tensor.device != torch.device(
        "cpu"
    ), "send tensor must be CUDA and dense"
    assert recv_tensor.device != torch.device(
        "cpu"
    ), "recv tensor must be CUDA and dense"

    if comm is None or comm is CommMember.WORLD:
        comm = _get_default_group().get_global_communicator()

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        comm.reduce(
            send_tensor.to_bagua_tensor().bagua_backend_tensor(),
            recv_tensor.to_bagua_tensor().bagua_backend_tensor(),
            dst,
            int(op),
        )

    comm.cuda_stream.synchronize()


def reduce_inplace(
    tensor: torch.Tensor, dst: int, op: ReduceOp = ReduceOp.SUM, comm: Optional[B.BaguaSingleCommunicatorPy] = None
):
    r"""The in-place version of :func:`reduce`."""

    if _rank_not_in_comm(comm):
        return

    assert tensor.device != torch.device("cpu"), "input tensor must be CUDA and dense"

    if comm is None or comm is CommMember.WORLD:
        comm = _get_default_group().get_global_communicator()

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        comm.reduce_inplace(
            tensor.to_bagua_tensor().bagua_backend_tensor(), dst, int(op)
        )

    comm.cuda_stream.synchronize()


def allreduce_coalesced_inplace(
    tensors,
    op: ReduceOp = ReduceOp.SUM,
    comm: Optional[B.BaguaSingleCommunicatorPy] = None,
):
    if _rank_not_in_comm(comm):
        return

    for tensor in tensors:
        assert tensor.device != torch.device(
            "cpu"
        ), "input tensors must be CUDA and dense"

    if comm is None or comm is CommMember.WORLD:
        comm = _get_default_group().get_global_communicator()

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        coalesced = flatten(tensors)
        comm.allreduce_inplace(
            coalesced.to_bagua_tensor("allreduce_coalesced"), int(op)
        )

        for buf, synced in zip(tensors, unflatten(coalesced, tensors)):
            buf.copy_(synced)

    # TODO: remove
    comm.cuda_stream.synchronize()


def allreduce(
    send_tensor: torch.Tensor,
    recv_tensor: torch.Tensor,
    op: ReduceOp = ReduceOp.SUM,
    comm: Optional[B.BaguaSingleCommunicatorPy] = None,
):
    """Reduces the tensor data across all processes associated with the communicator in such a way that all get
    the final result. After the call :attr:`recv_tensor` is going to be bitwise identical
    in all processes.

    Args:
        send_tensor: Input of the collective.
        recv_tensor: Output of the collective, must have the same size with :attr:`send_tensor`.
        op: One of the values from :class:`ReduceOp` enum. Specifies an operation used for element-wise reductions.
        comm: A handle of the Bagua communicator to work on. By default, the global
             communicator of the default process group will be used.

    Examples::

        >>> from bagua.torch_api import allreduce
        >>>
        >>> # All tensors below are of torch.int64 type.
        >>> # We have 2 process groups, 2 ranks.
        >>> send_tensor = torch.arange(2, dtype=torch.int64, device=tensor.device) + 1 + 2 * rank
        >>> recv_tensor = torch.zeros(2, dtype=torch.int64, device=tensor.device)
        >>> send_tensor
        tensor([1, 2], device='cuda:0') # Rank 0
        tensor([3, 4], device='cuda:1') # Rank 1
        >>> allreduce(send_tensor, recv_tensor)
        >>> recv_tensor
        tensor([4, 6], device='cuda:0') # Rank 0
        tensor([4, 6], device='cuda:1') # Rank 1

        >>> # All tensors below are of torch.cfloat type.
        >>> # We have 2 process groups, 2 ranks.
        >>> send_tensor = torch.tensor([1+1j, 2+2j], dtype=torch.cfloat, device=tensor.device) + 2 * rank * (1+1j)
        >>> recv_tensor = torch.zeros(2, dtype=torch.cfloat, device=tensor.device)
        >>> send_tensor
        tensor([1.+1.j, 2.+2.j], device='cuda:0') # Rank 0
        tensor([3.+3.j, 4.+4.j], device='cuda:1') # Rank 1
        >>> allreduce(send_tensor, recv_tensor)
        >>> recv_tensor
        tensor([4.+4.j, 6.+6.j], device='cuda:0') # Rank 0
        tensor([4.+4.j, 6.+6.j], device='cuda:1') # Rank 1
    """

    if _rank_not_in_comm(comm):
        return

    if comm is None or comm is CommMember.WORLD:
        comm = _get_default_group().get_global_communicator()

    assert send_tensor.device != torch.device(
        "cpu"
    ), "send tensor must be CUDA and dense"
    assert recv_tensor.device != torch.device(
        "cpu"
    ), "recv tensor must be CUDA and dense"


    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        comm.allreduce(
            send_tensor.to_bagua_tensor().bagua_backend_tensor(),
            recv_tensor.to_bagua_tensor().bagua_backend_tensor(),
            int(op),
        )

    # TODO: remove
    comm.cuda_stream.synchronize()


def allreduce_inplace(
    tensor: torch.Tensor,
    op: ReduceOp = ReduceOp.SUM,
    comm: Optional[B.BaguaSingleCommunicatorPy] = None,
):
    """The in-place version of :func:`allreduce`."""

    if _rank_not_in_comm(comm):
        return

    assert tensor.device != torch.device("cpu"), "input tensor must be CUDA and dense"

    if comm is None or comm is CommMember.WORLD:
        comm = _get_default_group().get_global_communicator()

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        comm.allreduce_inplace(tensor.to_bagua_tensor().bagua_backend_tensor(), int(op))

    comm.cuda_stream.synchronize()


def allgather(
    send_tensor: torch.Tensor,
    recv_tensor: torch.Tensor,
    comm: Optional[B.BaguaSingleCommunicatorPy] = None,
):
    """Gathers send tensors from all processes associated with the communicator into :attr:`recv_tensor`.

    Args:
        send_tensor (torch.Tensor): Input of the collective.
        recv_tensor (torch.Tensor): Output of the collective, must have a size of ``comm.nranks * send_tensor.size()`` elements.
        comm: A handle of the Bagua communicator to work on. By default, the global
             communicator of the default process group will be used.
    """

    if _rank_not_in_comm(comm):
        return

    assert send_tensor.device != torch.device(
        "cpu"
    ), "send tensor must be CUDA and dense"
    assert recv_tensor.device != torch.device(
        "cpu"
    ), "recv tensor must be CUDA and dense"

    if comm is None or comm is CommMember.WORLD:
        comm = _get_default_group().get_global_communicator()

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        comm.allgather(
            send_tensor.to_bagua_tensor().bagua_backend_tensor(),
            recv_tensor.to_bagua_tensor().bagua_backend_tensor(),
        )

    comm.cuda_stream.synchronize()


def allgather_inplace(
    tensor: torch.Tensor,
    comm: Optional[B.BaguaSingleCommunicatorPy] = None,
):
    """The in-place version of :func:`allgather`."""

    if _rank_not_in_comm(comm):
        return

    assert tensor.device != torch.device("cpu"), "input tensor must be CUDA and dense"

    if comm is None or comm is CommMember.WORLD:
        comm = _get_default_group().get_global_communicator()

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        comm.allgather_inplace(tensor.to_bagua_tensor().bagua_backend_tensor())

    comm.cuda_stream.synchronize()


def gather(
    send_tensor: torch.Tensor,
    recv_tensor: torch.Tensor,
    dst: int,
    comm: Optional[B.BaguaSingleCommunicatorPy] = None,
):
    """Gathers send tensors from all processes associated with the communicator to :attr:`recv_tensor` in a single process.

    Args:
        send_tensor: Input of the collective.
        recv_tensor: Output of the collective, must have a size of ``comm.nranks * send_tensor.size()`` elements.
        dst: Destination rank.
        comm: A handle of the Bagua communicator to work on. By default, the global
             communicator of the default process group will be used.
    """
    if _rank_not_in_comm(comm):
        return

    assert send_tensor.device != torch.device(
        "cpu"
    ), "send tensor must be CUDA and dense"
    assert recv_tensor.device != torch.device(
        "cpu"
    ), "recv tensor must be CUDA and dense"

    if comm is None or comm is CommMember.WORLD:
        comm = _get_default_group().get_global_communicator()

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        comm.gather(
            send_tensor.to_bagua_tensor().bagua_backend_tensor(),
            recv_tensor.to_bagua_tensor().bagua_backend_tensor(),
            dst,
        )

    comm.cuda_stream.synchronize()


def gather_inplace(
    tensor: torch.Tensor,
    count: int,
    dst: int,
    comm: Optional[B.BaguaSingleCommunicatorPy] = None,
):
    """The in-place version of :func:`gather`.

    Args:
        tensor: Input and output of the collective, On the :attr:`dst` rank, it
            must have a size of ``comm.nranks * count`` elements. On non-dst ranks, its size must
            be equal to :attr:``count``.
        count: The per-rank data count to gather.
        dst: Destination rank.
        comm: A handle of the Bagua communicator to work on. By default, the global
             communicator of the default process group will be used.
    """

    if _rank_not_in_comm(comm):
        return

    assert tensor.device != torch.device("cpu"), "input tensor must be CUDA and dense"

    if comm is None or comm is CommMember.WORLD:
        comm = _get_default_group().get_global_communicator()

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        comm.gather_inplace(tensor.to_bagua_tensor().bagua_backend_tensor(), count, dst)

    comm.cuda_stream.synchronize()


def scatter(
    send_tensor: torch.Tensor,
    recv_tensor: torch.Tensor,
    src: int,
    comm: Optional[B.BaguaSingleCommunicatorPy] = None,
):
    """Scatters send tensor to all processes associated with the communicator.

    Args:
        send_tensor: Input of the collective, must have a size of ``comm.nranks * recv_tensor.size()`` elements.
        recv_tensor: Output of the collective.
        src: Source rank.
        comm: A handle of the Bagua communicator to work on. By default, the global
             communicator of the default process group will be used.
    """

    if _rank_not_in_comm(comm):
        return

    assert send_tensor.device != torch.device(
        "cpu"
    ), "send tensor must be CUDA and dense"
    assert recv_tensor.device != torch.device(
        "cpu"
    ), "recv tensor must be CUDA and dense"

    if comm is None or comm is CommMember.WORLD:
        comm = _get_default_group().get_global_communicator()

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        comm.scatter(
            send_tensor.to_bagua_tensor().bagua_backend_tensor(),
            recv_tensor.to_bagua_tensor().bagua_backend_tensor(),
            src,
        )

    comm.cuda_stream.synchronize()


def scatter_inplace(
    tensor: torch.Tensor,
    count: int,
    src: int,
    comm: Optional[B.BaguaSingleCommunicatorPy] = None,
):
    """The in-place version of :func:`scatter`.

    Args:
        tensor: Input and output of the collective, On the :attr:`src` rank,
            it must have a size of ``comm.nranks * count`` elements. On non-src ranks,
            its size must be equal to :attr:`count`.
        count: The per-rank data count to scatter.
        src: Source rank.
        comm: A handle of the Bagua communicator to work on. By default, the global
             communicator of the default process group will be used.
    """

    if _rank_not_in_comm(comm):
        return

    assert tensor.device != torch.device("cpu"), "input tensor must be CUDA and dense"

    if comm is None or comm is CommMember.WORLD:
        comm = _get_default_group().get_global_communicator()

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        comm.scatter_inplace(
            tensor.to_bagua_tensor().bagua_backend_tensor(), count, src
        )

    comm.cuda_stream.synchronize()


def reduce_scatter(
    send_tensor: torch.Tensor,
    recv_tensor: torch.Tensor,
    op: ReduceOp = ReduceOp.SUM,
    comm: Optional[B.BaguaSingleCommunicatorPy] = None,
):
    """Reduces, then scatters :attr:`send_tensor` to all processes associated with the communicator.

    Args:
        send_tensor (torch.Tensor): Input of the collective, must have a size of ``comm.nranks * recv_tensor.size()`` elements.
        recv_tensor (torch.Tensor): Output of the collective.
        op (ReduceOp, optional): One of the values from :class:`ReduceOp` enum. Specifies an operation used for element-wise reductions.
        comm: A handle of the Bagua communicator to work on. By default, the global
             communicator of the default process group will be used.
    """

    if _rank_not_in_comm(comm):
        return

    assert send_tensor.device != torch.device(
        "cpu"
    ), "send tensor must be CUDA and dense"
    assert recv_tensor.device != torch.device(
        "cpu"
    ), "recv tensor must be CUDA and dense"

    if comm is None or comm is CommMember.WORLD:
        comm = _get_default_group().get_global_communicator()

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        comm.reduce_scatter(
            send_tensor.to_bagua_tensor().bagua_backend_tensor(),
            recv_tensor.to_bagua_tensor().bagua_backend_tensor(),
            int(op),
        )

    comm.cuda_stream.synchronize()


def reduce_scatter_inplace(
    tensor: torch.Tensor,
    op: ReduceOp = ReduceOp.SUM,
    comm: Optional[B.BaguaSingleCommunicatorPy] = None,
):
    """The in-place version of :func:`reduce_scatter`.

    Args:
        tensor (torch.Tensor): Input and output of the collective, the size must be divisible by ``comm.nranks``.
        op (ReduceOp, optional): One of the values from :class:`ReduceOp` enum. Specifies an operation used for element-wise reductions.
        comm: A handle of the Bagua communicator to work on. By default, the global
             communicator of the default process group will be used.
    """

    if _rank_not_in_comm(comm):
        return

    assert tensor.device != torch.device("cpu"), "send tensor must be CUDA and dense"

    if comm is None or comm is CommMember.WORLD:
        comm = _get_default_group().get_global_communicator()

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        comm.reduce_scatter_inplace(
            tensor.to_bagua_tensor().bagua_backend_tensor(), int(op)
        )

    comm.cuda_stream.synchronize()


def alltoall(
    send_tensor: torch.Tensor,
    recv_tensor: torch.Tensor,
    comm: Optional[B.BaguaSingleCommunicatorPy] = None,
):
    """
    Each process scatters :attr:`send_tensor` to all processes associated with the communicator and return the gathered
    data in :attr:`recv_tensor`.

    Args:
        send_tensor (torch.Tensor): Input of the collective, the size must be divisible by ``comm.nranks``.
        recv_tensor (torch.Tensor): Output of the collective, must have equal size with :attr:`send_tensor`.
        comm: A handle of the Bagua communicator to work on. By default, the global
             communicator of the default process group will be used.
    """
    if _rank_not_in_comm(comm):
        return

    assert send_tensor.device != torch.device(
        "cpu"
    ), "send tensor must be CUDA and dense"
    assert recv_tensor.device != torch.device(
        "cpu"
    ), "recv tensor must be CUDA and dense"

    if comm is None or comm is CommMember.WORLD:
        comm = _get_default_group().get_global_communicator()

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        comm.alltoall(
            send_tensor.to_bagua_tensor().bagua_backend_tensor(),
            recv_tensor.to_bagua_tensor().bagua_backend_tensor(),
        )

    comm.cuda_stream.synchronize()


# TODO combine **inplace API
def alltoall_inplace(
    tensor: torch.Tensor,
    comm: Optional[B.BaguaSingleCommunicatorPy] = None,
):
    """The in-place version of :func:`alltoall`."""
    if _rank_not_in_comm(comm):
        return

    assert tensor.device != torch.device("cpu"), "recv tensor must be CUDA and dense"

    if comm is None or comm is CommMember.WORLD:
        comm = _get_default_group().get_global_communicator()

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        comm.alltoall_inplace(tensor.to_bagua_tensor().bagua_backend_tensor())

    comm.cuda_stream.synchronize()


def alltoall_v(
    send_tensor: torch.Tensor,
    send_counts: int,
    send_displs: int,
    recv_tensor: torch.Tensor,
    recv_counts: int,
    recv_displs: int,
    comm: Optional[B.BaguaSingleCommunicatorPy] = None,
):
    """
    Each process scatters :attr:`send_tensor` to all processes associated with the communicator and return the gathered
    data in :attr:`recv_tensor`, each process may send a different amount of data and provide displacements for the input and output data.

    Args:
        send_tensor (torch.Tensor): Input of the collective, the size must be divisible by ``comm.nranks``.
        send_counts: integer array equal to the group size specifying the number of elements to send to each processor.
        send_displs: integer array (of length group size). Entry j specifies the displacement (relative to sendbuf from which to take the outgoing data destined for process j.
        recv_tensor (torch.Tensor): Output of the collective, must have equal size with :attr:`send_tensor`.
        recv_counts: integer array equal to the group size specifying the maximum number of elements that can be received from each processor.
        recv_displs: integer array (of length group size). Entry i specifies the displacement (relative to recvbuf at which to place the incoming data from process i.
        comm: A handle of the Bagua communicator to work on. By default, the global
             communicator of the default process group will be used.
    """
    if _rank_not_in_comm(comm):
        return

    assert send_tensor.device != torch.device(
        "cpu"
    ), "send tensor must be CUDA and dense"
    assert recv_tensor.device != torch.device(
        "cpu"
    ), "recv tensor must be CUDA and dense"

    if comm is None or comm is CommMember.WORLD:
        comm = _get_default_group().get_global_communicator()

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        comm.alltoall_v(
            send_tensor.to_bagua_tensor().bagua_backend_tensor(),
            send_counts,
            send_displs,
            recv_tensor.to_bagua_tensor().bagua_backend_tensor(),
            recv_counts,
            recv_displs,
        )

    comm.cuda_stream.synchronize()


def alltoall_v_inplace(
    tensor: torch.Tensor,
    counts: int,
    displs: int,
    comm: Optional[B.BaguaSingleCommunicatorPy] = None,
):
    """The in-place version of :func:`alltoall_v`."""
    if _rank_not_in_comm(comm):
        return

    assert tensor.device != torch.device("cpu"), "recv tensor must be CUDA and dense"

    if comm is None or comm is CommMember.WORLD:
        comm = _get_default_group().get_global_communicator()

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        comm.alltoall_v_inplace(tensor.to_bagua_tensor().bagua_backend_tensor(), counts, displs)

    comm.cuda_stream.synchronize()


def barrier(comm: Optional[B.BaguaSingleCommunicatorPy] = None):
    """
    Synchronizes all processes.
    This collective blocks processes until all processes associated with the
    communicator enters this function.

    Args:
        comm: A handle of the Bagua communicator to work on. By default, the global
             communicator of the default process group will be used.
    """
    if _rank_not_in_comm(comm):
        return

    if comm is None or comm is CommMember.WORLD:
        comm = _get_default_group().get_global_communicator()

    event = torch.cuda.current_stream().record_event()
    event.synchronize()

    with torch.cuda.stream(comm.cuda_stream):
        tensor = torch.ones([1], device=torch.cuda.current_device())
        comm.allreduce_inplace(tensor.to_bagua_tensor().bagua_backend_tensor(), op=ReduceOp.SUM)

    event = comm.cuda_stream.record_event()
    event.synchronize()
