import argparse
import sys


def get_args():
    ### Argument and global variables
    parser = argparse.ArgumentParser('TGAT')
    parser.add_argument('-d', '--data', type=str, choices=["wikipedia", "reddit", "alipay"],
                        help='Dataset name (eg. wikipedia or reddit)',
                        default='wikipedia')
    parser.add_argument('--tasks', type=str, default="LP", choices=["LP", "EC", "NC"],
                        help='task name link prediction, edge or node classification')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch_size')
    parser.add_argument('--emb_dimension', type=int, default=32, help='emb_dimension')
    parser.add_argument('--time_dimension', type=int, default=32, help='dimension of time-encoding')
    parser.add_argument('--prefix', type=str, default='TGAT', help='Prefix to name the checkpoints')
    parser.add_argument('--n_degree', type=int, default=20, help='Number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
    parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--n_layer', type=int, default=2, help='Number of network layers')
    parser.add_argument('--n_worker', type=int, default=0, help='Number of network layers')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight_decay')
    parser.add_argument('--seed', type=int, default=-1, help='Random Seed')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
    parser.add_argument('--warmup', action='store_true', help='')
    parser.add_argument('--edge_feat_dim', type=int, default=172, help='Dimensions of the edge feature')
    parser.add_argument('--node_feat_dim', type=int, default=32, help='Dimensions of the node feature')
    parser.add_argument('--uniform', action='store_true',
                        help='take uniform sampling from temporal neighbors')
    parser.add_argument('--balance', action='store_true',
                        help='use fraud user sampling when doing EC or NC tasks')
    parser.add_argument('--pretrain', action='store_true',
                        help='use linkpred task model as pretrain model to learn EC or NC')
    parser.add_argument('--no_time', action='store_true',
                        help='do not use time embedding')
    parser.add_argument('--no_pos', action='store_true',
                        help='do not use position embedding')
    parser.add_argument("--nu", type=float, default=0.2,
            help="hyperparameter nu (must be 0 < nu <= 1)")

    try:
        args = parser.parse_args()
        assert args.n_worker == 0, "n_worker must be 0, etherwise dataloader will cause bug and results very bad performance (this bug will be fixed soon)"
        if args.data == 'alipay':
            args.feat_dim = 101
            args.lr *= 5
            args.bs *= 5
        else:
            args.feat_dim = 172
        args.no_time = True
        # args.no_pos = True

    except:
        parser.print_help()
        sys.exit(0)

    return args

