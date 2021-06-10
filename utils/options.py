def add_opts(parser):
    # options for dataset
    parser.add_argument("--HO3D_root", type=str, help="HO3D dataset root", required=True)
    parser.add_argument("--mano_root", default="assets/mano_models", type=str, help="mano root")
    parser.add_argument("--obj_model_root", default="assets/object_models", type=str, help="object model root")
    parser.add_argument("--inp_res", default=512, type=int, help="input image size")

    # options for model
    parser.add_argument("--network", default="honet_transformer",
                        choices=["honet_attention", "honet_transformer"], help="network architecture")
    parser.add_argument('-s', '--stacks', default=1, type=int,
                        help='Number of hourglasses to stack (default: 1)')
    parser.add_argument('--channels', default=256, type=int,
                        help='Number of channels in the hourglass (default: 256)')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass (default: 1)')

    # options for loss
    parser.add_argument("--mano_neurons", nargs="+", default=[1024, 512], type=int,
                        help="Number of neurons in hidden layer for mano decoder")
    parser.add_argument("--mano_lambda_joints3d", default=1e4, type=float,
                        help="Weight to supervise joints in 3d")
    parser.add_argument("--mano_lambda_verts3d", default=1e4, type=float,
                        help="Weight to supervise vertices in 3d")
    parser.add_argument("--mano_lambda_manopose", default=10, type=float,
                        help="Weight to supervise mano pose parameters")
    parser.add_argument("--mano_lambda_manoshape", default=0.1, type=float,
                        help="Weight to supervise mano shape parameters")
    parser.add_argument("--mano_lambda_regulshape", default=1e2, type=float,
                        help="Weight to regularize hand shapes")
    parser.add_argument("--mano_lambda_regulpose", default=1, type=float,
                        help="Weight to regularize hand pose in axis-angle space")
    parser.add_argument("--lambda_joints2d", default=1e2, type=float, help="Weight to supervise joints in 2d")
    parser.add_argument("--lambda_objects", default=5e2, type=float, help="Weight to supervise objects")
    parser.add_argument("--transformer_depth", default=1, type=int, help="transformer module depth")
    parser.add_argument("--transformer_head", default=1, type=int, help="transformer attention head")

    # options for training
    parser.add_argument("--manual_seed", default=0, type=int, help="manual seed")
    parser.add_argument("-j", "--workers", default=16, type=int, help="number of data loading workers (default: 16)")
    parser.add_argument("--epochs", default=60, type=int, help="number of total epochs to run")
    parser.add_argument("--train_batch", default=24, type=int, help="Train batch size")
    parser.add_argument("--test_batch", default=16, type=int, metavar="N", help="Test batch size")
    parser.add_argument("--lr", "--learning-rate", default=1e-4, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--lr_decay_step", nargs="+", default=10, type=int, help="epochs after which to decay learning rate")
    parser.add_argument("--lr_decay_gamma", default=0.7, type=float, help="factor by which to decay the learning rate")
    parser.add_argument("--weight_decay", default=0.0005, type=float)

    # options for exp
    parser.add_argument("--host_folder", default="./exp-results",
                        type=str, help="path to save experiment results")
    parser.add_argument("--resume", type=str, help="path to latest checkpoint")
    parser.add_argument("--evaluate", dest="evaluate", action="store_true",
                        help="evaluate model")
    parser.add_argument("--save_results", default=0, help="save output results of the network")
    parser.add_argument("--test_freq", type=int, default=None,
                        help="testing frequency on evaluation dataset (set specific in traineval.py)")
    parser.add_argument("--snapshot", default=10, type=int, metavar="N",
                        help="How often to take a snapshot of the model (0 = never)")
    parser.add_argument("--use_cuda", default=1, type=int, help="use GPU (default: True)")
