# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib




import argparse
import os
import statistics
import json
import torch
import tqdm
import platform
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
import src.hypes_yaml.yaml_utils as yaml_utils
from src.tools import train_utils
from src.tools import multi_gpu_utils
from src.data_utils.datasets import build_dataset
from src.tools import train_utils
import warnings
import time
import numpy as np
warnings.filterwarnings("ignore")




def train_parser():

    

    """
    Initializes and returns a command-line argument parser for the training process.

    Usage examples:
        python train.py --hypes_yaml [path_to_yaml]
        python train.py --hypes_yaml [path_to_yaml] --half
    
    Resturn:
        opt: It will be something like 
        Namespace(dist_url='env://', half=False, hypes_yaml='src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml', model_dir='')
    
    done
    """
    # Create an argument parser with a brief description
    parser = argparse.ArgumentParser(description="Parser for training with synthetic data")

    # Required: Path to the YAML configuration file for hyperparameters
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help="Path to the YAML configuration file for training")

    # Optional: Directory for saving or loading model checkpoints
    parser.add_argument('--model_dir', default='',
                        help="Directory to load/save the model checkpoint")

    # Optional: Enable mixed-precision training (half-precision)
    parser.add_argument("--half", action='store_true',
                        help="Use half-precision training to reduce memory consumption")

    # Optional: Distributed training initialization URL (default uses environment variable)
    parser.add_argument('--dist_url', default='env://',
                        help="URL for initializing distributed training")

    # Parse and return the command-line arguments
    opt = parser.parse_args()
    return opt


start_time = time.time()
def main():
    num_workers = 4
    prefetch_factor = 4
    opt = train_parser()
    print("*********************Step0: Train parser completed *********************")
    print('You passed the following options:\n',opt)
    
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    print("*********************Step1: Yaml file Readed*********************")
    print("The configuration setup read from above path is as follow:\n",hypes)

    
    print('*********************Step2: Multi GPU Checking*********************',end="\n")
    multi_gpu_utils.init_distributed_mode(opt)
    

    print('*********************Step3_1: Train Dataset Building*********************')
    src_train_dataset = build_dataset(dataset_cfg=hypes, visualize=False, train=True)
    print('*********************Step3_2: Validate Dataset Building*********************')
    src_validate_dataset = build_dataset(dataset_cfg=hypes, visualize=False, train=False)
    
    if opt.distributed:
        print('*********************Step4: DataLoader Creating*********************')
        print('Since Distributed training environment detected. Initializing DistributedSampler for data parallelization is done.')
        sampler_train = DistributedSampler(src_train_dataset)
        sampler_val = DistributedSampler(src_validate_dataset,
                                         shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        print(f"Numnber of workers was set to:{num_workers}")
        train_loader = DataLoader(src_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=num_workers, # These were 8, due to error I set them to 0
                                  collate_fn=src_train_dataset.collate_batch_train)
        val_loader = DataLoader(src_validate_dataset,
                                sampler=sampler_val,
                                num_workers=num_workers,
                                collate_fn=src_train_dataset.collate_batch_train,
                                drop_last=False)
    else:
        print('*********************Step4: DataLoader Creating*********************')
        print('Since Distributed training environment not detected. Initializing DataLoader for data parallelization is done.')
        print(f"Numnber of workers was set to:{num_workers}")
        train_loader = DataLoader(src_train_dataset,
                                  batch_size=hypes['train_params']['batch_size'],
                                  num_workers=num_workers,
                                  collate_fn=src_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True,
                                  prefetch_factor=prefetch_factor)
        val_loader = DataLoader(src_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=num_workers,
                                collate_fn=src_train_dataset.collate_batch_train,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=True,
                                prefetch_factor=prefetch_factor)

    print('*********************Step5: Creating Model*********************')
    model = train_utils.create_model(hypes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Model will run ({device}) device.')
    # if we want to train from last checkpoint.
    if opt.model_dir:
        print(f'Model directory {opt.model_dir} detected. Initializing model from saved checkpoint.')
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path,
                                                         model)

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module

    # define the loss
    print('*********************Step6: Creating Loss Function*********************')
    criterion = train_utils.create_loss(hypes)
    print("Conf Loss = Confidence loss = Classification loss * Classification_weight")
    print("Loc Loss = reg_loss")
    print("Total loss = reg_loss + conf_loss")

    # optimizer setup
    print('*********************Step7: Creating Optimizer*********************')
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('*********************Step8: Starting Training part*********************')
    epoches = hypes['train_params']['epoches']
    batch_size = hypes['train_params']['batch_size']
    # used to help schedule learning rate
    print(f"Batch_size = {batch_size}")
    print(f"Number of epochs = {epoches}")
    print(f"Number of batch_Data to analyse in each epoch = {len(train_loader)}")

    
    loss_dict = {}
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        st = time.time()

        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            print('-'*25)
            print()
            print(f"For Epoch {epoch}:")
            print('learning rate is %.7f ' % param_group["lr"])
            print(' || ')
            print(" || ")
            print(' ** ')
        if opt.distributed:
            sampler_train.set_epoch(epoch)
        writer.add_scalar("LR/epoch", param_group["lr"], epoch)
        writer.flush()
        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)
        

        index =0
        train_loss_batch=[]
        print("Training Epoch %d .........." % epoch)   
        for batch_data in train_loader:

            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    # if epoch == 0:
                    #     make_dot(ouput_dict, params=dict(model.named_parameters())).render("model_graph", format="png")
                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])

            
            criterion.logging(epoch, index, len(train_loader), writer, pbar=pbar2)
            pbar2.update(1)
       
            loss_value = final_loss.item()


            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + index)
            train_loss_batch.append(loss_value)
            writer.add_scalar('Train_Loss/batch', loss_value,
                              epoch * len(train_loader) + index)
            writer.flush()
            index +=1


        print(f"\nAt epoch {epoch}, the mean train loss is: {np.mean(train_loss_batch)}")
        print(f"At epoch {epoch}, the min train loss is: {np.min(train_loss_batch)}")
        
        writer.add_scalar('Train_Loss/epoch', np.mean(train_loss_batch), epoch)
        writer.flush()

   

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            

            with torch.no_grad():
                valid_loss_batch = []
                print('Validating epoch %d ..........' % epoch)
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_loss_batch.append(final_loss.item())
                    writer.add_scalar('Validate_Loss/batch',
                                      final_loss.item(),
                                      epoch * len(val_loader) + i)
                    writer.flush()

            print('At epoch %d, the mean validation loss is: %f' % (epoch,
                                                              np.mean(valid_loss_batch)))
            print('At epoch %d, the min validation loss is: %f' % (epoch,
                                                              np.min(valid_loss_batch)))
            writer.add_scalar('Validate_Loss/epoch', np.mean(valid_loss_batch), epoch)
            writer.flush()

        loss_dict[epoch] = {
            'train_loss': np.mean(train_loss_batch),
            'train_loss_min': np.min(train_loss_batch),
            'train_loss_batch': train_loss_batch,
            'val_loss': np.mean(valid_loss_batch) if 'valid_loss_batch' in locals() else None,
            'val_loss_min': np.min(valid_loss_batch) if 'valid_loss_batch' in locals() else None,
            'val_loss_batch': valid_loss_batch if 'valid_loss_batch' in locals() else None
        }
        with open(os.path.join(saved_path, f'loss_dict_epoch_{epoch}.json'), 'w') as f:
            json.dump(loss_dict, f, indent=4)

        sp = time.time()
        print(f"Total training time for epoch {epoch} : {((sp - st)/60)} minutes")
        print('-'*50)

    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
end_time = time.time()
print(f"Total training time: {((end_time - start_time)/60)} minutes")