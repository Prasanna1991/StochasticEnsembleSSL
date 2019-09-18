import argparse

parser = argparse.ArgumentParser('StochasticEnsembleSSL')

parser.add_argument('--data_root', type=str, default='../../data/')
parser.add_argument('--random_seed', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch', type=int, default=128, help='Batch size to use')
parser.add_argument('--nz', type=int, default=60)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--n_classes', type=int, default=14)
parser.add_argument('--mode', type=float, default=3,
                    help='1 --> train VAE, 2 --> train ensemble SSL, 3 --> test')


#config for ensembling
parser.add_argument('--alpha', type=float, default=0.6)
parser.add_argument('--cut_off_epoch', type=float, default=220.0)
parser.add_argument('--cut_off_value', type=float, default=0.10)
parser.add_argument('--ramp_up_mult', type=float, default=-5.0)
parser.add_argument('--epochs_ensemble', type=int, default=4000)
parser.add_argument('--max_epochs', type=int, default=1000)
parser.add_argument('--max_val', type=int, default=30)

config, _ = parser.parse_known_args()