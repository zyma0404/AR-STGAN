import argparse
import torch
import numpy as np
import random
import os
from trainer import Trainer
from tester import Tester

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='bay', help='bay or nyc')
parser.add_argument('--root_path', type=str, default='./AR-STGAN/', help='root path: dataset, checkpoint')

parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension.')
parser.add_argument('--epoch', type=int, default=30, help='Number of training epochs per iteration.')
parser.add_argument('--batch_size', type=int, default=256)  

parser.add_argument('--lambda_G', type=int, default=500, help='lambda_G for generator loss function')
parser.add_argument('--embed_dim', type=int, default=10, help='the dimension of node embedding')
parser.add_argument('--cheb_k', type=int, default=3)
parser.add_argument('--default_graph', type=eval, default=True)
parser.add_argument('--seq_len', type=int, default=2016)

parser.add_argument('--num_adj', type=int, default=9, help='number of nodes in sub graph')
parser.add_argument('--num_layer', type=int, default=2, help='number of layers in LSTM and DCRNN')
parser.add_argument('--trend_time', type=int, default=7 * 24, help='the length of trend segment is 7 days')

parser.add_argument('--cpu', type=bool, default=False)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cuda_id', type=str, default='0')
parser.add_argument('--seed', type=int, default=20)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# parameter
opt = vars(args)

if opt['dataset'] == 'bay':
    opt['timestamp'] = 12  # 5min: 12 or 30min: 2
    opt['train_time'] = 105  # days for training
    opt['recent_time'] = 1  # bay: 1 hour, nyc: 2hour
    opt['num_feature'] = 6 * 2  # length of input feature  * 2
    opt['time_feature'] = 31  # length of time feature
    opt['num_node'] = 365

elif opt['dataset'] == 'nyc':
    opt['timestamp'] = 2  # 5min: 12 or 30min: 2
    opt['train_time'] = 289  # days for training
    opt['recent_time'] = 2  # bay: 1 hour, nyc: 2hour
    opt['num_feature'] = 2 * 2  # length of input feature
    opt['time_feature'] = 39  # length of time feature
    opt['num_node'] = 862

opt['save_path'] = opt['root_path'] + opt['dataset'] + '/checkpoint/'
opt['data_path'] = opt['root_path'] + opt['dataset'] + '/data/'
opt['result_path'] = opt['root_path'] + opt['dataset'] + '/result/'
opt['train_time'] = opt['train_time'] * opt['timestamp'] * 24

if __name__ == "__main__":
    print("Starting training...")
    opt['isTrain'] = True
    train_model = Trainer(opt)
    best_epoch = train_model.train()

    # print("Starting testing...")
    # opt['isTrain'] = False
    # test_model = Tester(opt)
    # test_model.test()
    for epoch in range(best_epoch+5):
        print(f"Starting testing epoch = {epoch+1}")
        opt['isTrain'] = False
        opt['epoch'] = epoch + 1
        test_model = Tester(opt)
        test_model.test()












