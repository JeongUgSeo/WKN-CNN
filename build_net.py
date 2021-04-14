import torch
import torch.nn as nn
from net import *

def build_net(net, cfg, shape, load_path, gpu):
    print("[Building Net...]")
    # Build Net
    if net == 'CNN':
        net = CNN(shape)

    if net == 'EEGNet2D':
        net = EEGNet2D(shape, cfg)

    if net == 'Seo_CNN':
        net = Seo_CNN(shape, cfg)

    # Load net NOTE: 고쳐야 함
    if load_path:
        print("[Loading Net...]")
        # checkpoint = torch.load('./comparison/Cram_not_bi.tar')
        # target_params = {k:v for k, v in checkpoint['net_state_dict'].items() if k.split(".")[0] != 'cnn'}
        # net.load_state_dict(target_params, strict=False)
        # load_path = args.pretrained or args.train_cont
        # if load_path:
        #     print("Loading net...")
        #     checkpoint = torch.load(args.load_path)
        #     pretrained_epoch = checkpoint['epoch']
        #     net.load_state_dict(checkpoint['net_state_dict'])

    # Set GPU
    if gpu != 'cpu':
        assert torch.cuda.is_available(), "Check GPU"
        if gpu == 'multi':
            device = torch.device
            net = nn.DataParallel(net)
        else:
            device = torch.device(f'cuda:{gpu}')
            torch.cuda.set_device(device)
        net.cuda()

    # Set CPU
    else:
        device = torch.device("cpu")
    # Print
    print(f"device: {device}")

    return net