import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser()

    # select gpu
    parser.add_argument('--cuda', type=int, default=1)
    
    # fit the data
    parser.add_argument('--time', type=int, default=100)
    parser.add_argument('--neurons', type=int, default=100)

    # hyper parameters
    parser.add_argument('--dim_e', type=int, default=16)
    parser.add_argument('--dim_c', type=int, default=32)
    parser.add_argument('--dim_d', type=int, default=32)
    parser.add_argument('--gcn_hidden', type=list, default=[32, 8])
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--time_lag', type=int, default=0)
    parser.add_argument('--dropout_prob', type=float, default=0.8)
    parser.add_argument('--kl_weight_1', type=float, default=1)
    parser.add_argument('--kl_weight_2', type=float, default=0.1)
    parser.add_argument('--autocorrelation_taus', type=float, default=10.0)
    parser.add_argument('--noise_variances', type=float, default=0.1)
    parser.add_argument('--dynamic_graph', type=bool, default=False)
    parser.add_argument('--discrete_graph', type=bool, default=True)
    
    # training strategy
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.1)    #0.001 
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--gamma', type=float, default=0.999)

    # training managing
    parser.add_argument('--model_name', type=str, default=time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())))
    parser.add_argument('--model_path', type=str, default=None)

    args = parser.parse_args()
    return args