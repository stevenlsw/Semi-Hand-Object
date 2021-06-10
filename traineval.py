import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim

from utils.utils import Monitor, get_dataset, get_network, print_args, save_args, load_checkpoint, save_checkpoint
from utils.epoch import single_epoch
from utils.options import add_opts


def main(args):
    # Initialize randoms seeds
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    # create exp result dir
    os.makedirs(args.host_folder, exist_ok=True)
    # Initialize model
    model = get_network(args)

    if args.use_cuda and torch.cuda.is_available():
        print("Using {} GPUs !".format(torch.cuda.device_count()))
        model.cuda()

    start_epoch = 0
    if not args.evaluate:
        model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(model_params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, gamma=args.lr_decay_gamma)
        train_dat = get_dataset(args, mode="train")
        print("training dataset size: {}".format(len(train_dat)))
        train_loader = torch.utils.data.DataLoader(train_dat, batch_size=args.train_batch, shuffle=True,
                                                   num_workers=int(args.workers), pin_memory=True, drop_last=False)
        monitor = Monitor(hosting_folder=args.host_folder)

    else:
        assert args.resume is not None, "need trained model for evaluation"
        device = torch.device('cuda')if torch.cuda.is_available() and args.use_cuda else torch.device('cpu')
        load_checkpoint(model, resume_path=args.resume, strict=False, device=device)
        args.epochs = start_epoch + 1

    # Initialize validation dataset
    val_dat = get_dataset(args, mode="evaluation")
    print("evaluation dataset size: {}".format(len(val_dat)))
    val_loader = torch.utils.data.DataLoader(val_dat, batch_size=args.test_batch,
                                             shuffle=False, num_workers=int(args.workers),
                                             pin_memory=True, drop_last=False)

    for epoch in range(start_epoch, args.epochs):
        train_dict = {}
        if not args.evaluate:
            print("Using lr {}".format(optimizer.param_groups[0]["lr"]))
            train_avg_meters = single_epoch(
                loader=train_loader, model=model, optimizer=optimizer,
                epoch=epoch, save_path=args.host_folder, train=True,
                save_results=False, use_cuda=args.use_cuda)

            train_dict = {meter_name: meter.avg
                          for (meter_name, meter) in train_avg_meters.average_meters.items()}
            monitor.log_train(epoch + 1, train_dict)

        # Evaluate on validation set
        if args.evaluate or (epoch + 1) % args.test_freq == 0:
            with torch.no_grad():
                single_epoch(loader=val_loader, model=model, epoch=epoch if not args.evaluate else None,
                             optimizer=None, save_path=args.host_folder,
                             train=False, save_results=args.save_results, use_cuda=args.use_cuda,
                             indices_order=val_dat.jointsMapSimpleToMano if hasattr(val_dat, "jointsMapSimpleToMano") else None)

        if not args.evaluate:
            if (epoch+1) % args.snapshot == 0:
                print(f"save epoch {epoch+1} checkpoint to {args.host_folder}")
                save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "network": args.network,
                    "state_dict": model.state_dict(),
                },
                checkpoint=args.host_folder, filename=f"checkpoint_{epoch+1}.pth.tar")

            if args.lr_decay_gamma:
                if args.lr_decay_step is None:
                    scheduler.step(train_dict["mano_joints3d_loss"])
                else:
                    scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand-Object training")
    add_opts(parser)

    args = parser.parse_args()
    args.test_freq = 10
    args.save_results = True
    args.snapshot = 10

    print_args(args)
    save_args(args, save_folder=args.host_folder, opt_prefix="option")
    main(args)
    print("All done !")