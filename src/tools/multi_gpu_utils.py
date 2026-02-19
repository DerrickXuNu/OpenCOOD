import os
import torch
import torch.distributed as dist

def init_distributed_mode(args):
    """
    Initialize the distributed training environment.

    This function supports various distributed launch methods such as:
    - PyTorch native launch via `torch.distributed.launch`
    - SLURM cluster-based job scheduling
    
    It sets the GPU device, initializes the process group, and synchronizes processes.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments including `dist_url`, `rank`, `world_size`, etc.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Standard PyTorch multi-GPU launch
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif 'SLURM_PROCID' in os.environ:
        # SLURM cluster-based distributed launch
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print(
            'No supported multi-GPU setup found.\n'
            'If you are using a multi-GPU system, please ensure '
            'distributed environment variables are properly configured.'
        )
        args.distributed = False
        return

    args.distributed = True
    args.dist_backend = 'nccl'

    # Set the current process to the correct GPU
    torch.cuda.set_device(args.gpu)

    # Print initialization info for debugging
    print(f'| Distributed initialization (rank {args.rank}): {args.dist_url}', flush=True)
    
    # Verify GPU setup
    print(f'| GPU device set: {args.gpu}', flush=True)
    print(f'| Current CUDA device: {torch.cuda.current_device()}', flush=True)
    print(f'| CUDA device name: {torch.cuda.get_device_name(args.gpu)}', flush=True)
    print(f'| Total GPUs available: {torch.cuda.device_count()}', flush=True)

    # Initialize the default process group
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )

    # Synchronize all processes
    dist.barrier()

    # Disable printing in non-master processes
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    Override the built-in print function to suppress output from non-master processes.

    Parameters
    ----------
    is_master : bool
        If True, printing is enabled. Otherwise, it's disabled unless 'force=True' is passed.
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
