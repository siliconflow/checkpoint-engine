import argparse
import json
import os
import pickle
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from typing import Literal

import httpx
import torch
import zmq
from torch.multiprocessing import Queue, get_context
import torch.distributed as dist
from loguru import logger
from safetensors import safe_open

from checkpoint_engine.ps import ParameterServer, _get_physical_gpu_id
from checkpoint_engine.worker import update_weights_from_ipc
from checkpoint_engine.device_utils import DeviceManager
device_manager = DeviceManager()

@contextmanager
def timer(msg: str):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    logger.info(f"{msg} duration: {end - start:.2f} seconds")

def split_tensors_by_layer(checkpoint_path: str, rank: int, world_size: int, num_layers: int, layer_prefix='model.layers.') -> dict[str, torch.Tensor]:
    index_fn = os.path.join(checkpoint_path, "model.safetensors.index.json")
    with open(index_fn) as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]
    curr_layers = []# layer_idx[num_layers] save other parameters
    for layer_idx in range(rank, num_layers+1, world_size):
        curr_layers.append(layer_idx)
    others_map = {}
    packaged_items: dict[int, dict[str, list[str]]] = {}
    for layer_idx in curr_layers:
        packaged_items[layer_idx] = {} # dict[str, list[str]]
    fn_tensors: dict[str, list[str]] = defaultdict(list)
    for name, file in weight_map.items():
        if name.startswith(layer_prefix):
            layer_num_str = name[len(layer_prefix):].split('.')[0]
            try:
                layer_num = int(layer_num_str)
                if layer_num not in curr_layers:
                    continue
                fn_tensors[file].append(name)
                if "attn" in name:
                    packaged_items[layer_num].setdefault("attn", []).append(name)
                elif "mlp.experts" in name:
                    if "gate_proj" in name or "up_proj" in name:
                        packaged_items[layer_num].setdefault("experts_gate_up", []).append(name)
                    else:
                        packaged_items[layer_num].setdefault("experts_down", []).append(name)
                elif "mlps" in name:
                    packaged_items[layer_num].setdefault("mlp", []).append(name)
                else:
                    packaged_items[layer_num].setdefault("others", []).append(name)
            except ValueError:
                others_map[name] = file
        else:
            others_map[name] = file
    # save other parameters
    if num_layers in curr_layers:
        packaged_items[num_layers] = {}
        for name, file in others_map.items():
            fn_tensors[file].append(name)
            packaged_items[num_layers].setdefault("others", []).append(name)
    named_tensors = {}
    for file, names in fn_tensors.items():
        with safe_open(os.path.join(checkpoint_path, file), framework="pt") as f:
            for name in names:
                named_tensors[name] = f.get_tensor(name)
    
    target_size = len(weight_map.items())
    curr_rank_size = len(named_tensors)
    packaged_items_size = sum(len(v) for item in packaged_items.values() for v in item.values())
    logger.info(f"rank {rank} {curr_layers=} {curr_rank_size=}, {packaged_items_size=}, {target_size=}")
    assert curr_rank_size == packaged_items_size, f"rank {rank} loaded tensors size {curr_rank_size} not equal to packaged_items tensors size {packaged_items_size}"
    device = f'cuda:{os.getenv("LOCAL_RANK", 0)}'
    tmp_tensor = torch.tensor([curr_rank_size], dtype=torch.int64, device=device)
    dist.init_process_group(rank=rank, world_size=world_size)
    dist.barrier()
    dist.all_reduce(tmp_tensor, op=dist.ReduceOp.SUM)
    dist.destroy_process_group()
    total_size = tmp_tensor.item()
    assert total_size == target_size, f"{rank} all ranks loaded tensors size {total_size} not equal to target size {target_size}"
    return named_tensors, packaged_items


def checker_proc(rank: int, device_uuid: str, model_path: str, num_layers: int, queue: Queue):
    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    named_tensors, packaged_infos = split_tensors_by_layer(model_path, rank, world_size, num_layers=num_layers, layer_prefix='model.layers.')
    torch.cuda.set_device(rank)
    named_tensors = {name: tensor.cuda() for name, tensor in named_tensors.items()}
    _zmq_ctx = zmq.Context()

    def check(names_to_check: dict[str, bool], weights: list[tuple[str, torch.Tensor]]):
        for name, weight in weights:
            if name not in named_tensors:
                continue
            assert (weight == named_tensors[name]).all()
            names_to_check[name] = True

    def check_weights(names_to_check: dict[str, bool], socket_paths: list[tuple[str, str]]):
        socket_paths = dict(socket_paths)
        update_weights_from_ipc(
            _zmq_ctx,
            socket_paths[device_uuid],
            device_id=rank,
            run=lambda weights: check(names_to_check, weights),
            post_hook=lambda: torch.cuda.synchronize(),
        )
        assert all(names_to_check.values())

    while True:
        socket_paths: list[tuple[str, str]] = queue.get()
        if socket_paths is None:
            break
        names_to_check = dict.fromkeys(named_tensors.keys(), False)
        check_weights(names_to_check, socket_paths)


def register_weights(
    ps: ParameterServer,
    checkpoint_name: str,
    checkpoint_files: list[str],
    named_tensors: dict[str, torch.Tensor],
    packaged_infos: dict[int, dict[str, list[str]]] | None = None,
    save_metas_file: str | None = None,
):
    ps.register_checkpoint(
        checkpoint_name,
        files=checkpoint_files,
        named_tensors=named_tensors,
        use_shared_memory_pool=False,
        use_inplace_pin_memory=False,
        packaged_infos=packaged_infos)
    ps.init_process_group()
    dist.barrier()
    with timer("Gather metas"):
        ps.gather_metas(checkpoint_name)
    if save_metas_file and int(os.getenv("RANK")) == 0:
        with open(save_metas_file, "wb") as f:
            pickle.dump(ps.get_metas(), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update weights example")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--save-metas-file", type=str, default=None)
    parser.add_argument("--sleep-time", type=int, default=3600)
    parser.add_argument("--checkpoint-name", type=str, default=None)
    args = parser.parse_args()
    if args.checkpoint_name is None:
        args.checkpoint_name = os.path.basename(args.checkpoint_path.rstrip("/"))
    if args.save_metas_file is None:
         args.save_metas_file = os.path.basename(args.checkpoint_path.rstrip("/")) + "_ckpt_metas.pkl"
    logger.info("=============== Arguments ===============")
    logger.info(f"{args.checkpoint_path=}")
    logger.info(f"{args.num_layers=}")
    logger.info(f"{args.checkpoint_name=}")
    logger.info(f"{args.save_metas_file=}")
    logger.info(f"{args.sleep_time=}")
    logger.info("=========================================")
    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    ctx = get_context("spawn")
    queue = ctx.Queue()
    _device_uuid = _get_physical_gpu_id(device_manager, rank)
    ps = ParameterServer(auto_pg=True)
    assert os.path.exists(os.path.join(args.checkpoint_path, "model.safetensors.index.json")), f"index file not found in {args.checkpoint_path}"
    named_tensors, packaged_infos = split_tensors_by_layer(args.checkpoint_path, rank, world_size, num_layers=args.num_layers, layer_prefix='model.layers.')
    logger.info(f"rank {rank} loaded tensors size {len(named_tensors)}, packaged items size {len(packaged_infos)}")
    # for layer_idx, items in packaged_infos.items():
    #     logger.info(f"layer {layer_idx}:")
    #     for item_name, tensors in items.items():
    #         logger.info(f"  {item_name}: {len(tensors)} tensors")
    proc = ctx.Process(target=checker_proc, args=(rank, _device_uuid, args.checkpoint_path, args.num_layers, queue))
    proc.start()
    logger.info(f"rank {rank} start registering checkpoint")
    checkpoint_files = []
    register_weights(
        ps,
        args.checkpoint_name,
        checkpoint_files,
        named_tensors,
        packaged_infos,
        args.save_metas_file,
    )
    logger.info(f"rank {rank} finished registering checkpoint")
    ps.update(args.checkpoint_name, queue.put, ranks=[], buffer_granularity=True)
    logger.info(f"rank {rank} finished updating weights")

    # ps.unregister_checkpoint(args.checkpoint_name)
    queue.put(None)
    proc.join()
    logger.info(f"rank {rank} finished checking weights")
    time.sleep(args.sleep_time)
