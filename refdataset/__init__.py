from .transform import get_transform as get_transform_seg
from .refdataset import Refdataset, collate_fn

def build_dataset(is_train, args):

    data_path = args.data_path
    if is_train:
        split = 'train'
    else:
        split = args.test_split
    if args.data_set == 'refcoco':
        dataset = Refdataset(refer_data_root=data_path, dataset='refcoco', splitBy='unc', split=split, image_transforms=get_transform_seg((args.input_size, args.input_size)))
    elif args.data_set == 'refcoco+':
        dataset = Refdataset(refer_data_root=data_path, dataset='refcoco+', splitBy='unc', split=split, image_transforms=get_transform_seg((args.input_size, args.input_size)))
    elif args.data_set == 'refcocog':
        dataset = Refdataset(refer_data_root=data_path, dataset='refcocog', splitBy='umd', split=split, image_transforms=get_transform_seg((args.input_size, args.input_size)))
    elif args.data_set == 'arefcoco':
        dataset = Refdataset(refer_data_root=data_path, dataset='arefcoco', splitBy=None, split=split, image_transforms=get_transform_seg((args.input_size, args.input_size)))
    else:
        raise

    return dataset

def build_dataloader(args):
    """
    Build training and validation dataloaders.
    
    Args:
        args: Arguments containing dataset configuration
        
    Returns:
        tuple: (data_loader_train, data_loader_val)
    """
    import torch
    
    # Build datasets
    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)
    
    # Create samplers
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # Create dataloaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        collate_fn=collate_fn,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=collate_fn,
    )
    
    return data_loader_train, data_loader_val

