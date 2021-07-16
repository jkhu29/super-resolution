import argparse


def get_options():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--valid_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='/content')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers, you had better put it '
                                                               '4 times of your gpu')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size, default=32')
    parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for, default=10')
    parser.add_argument('--lr', type=float, default=8e-4, help='select the learning rate, default=2e-4')
    parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
    parser.add_argument('--adam', action='store_true', default=False, help='whether to use adam')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
    parser.add_argument('--seed', type=int, default=118, help="whether to continue training")
    opt = parser.parse_args()

    return opt

