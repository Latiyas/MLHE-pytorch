import os, sys, pickle, time, torch, random, utils
import argparse
from monotonelearning import MonotoneLearning


"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of monotone learning"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--algo_type', type=str, default='retraining',
                        choices=['restarttraining', 'retraining', 'MTsimple', 'MLHE'],
                        help='The type of algorithm')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        choices=['MNIST', 'CIFAR10', 'SST2', 'TinyImageNet'],
                        help='The name of dataset')
    parser.add_argument('--divide_type', type=int, default=4, choices=[3, 4, 5, 6],
                        help='The type of division')
    parser.add_argument('--output_model', type=int, default=0, choices=[0, 1], help='The output model')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--save_dir', type=str, default='../user_data/model_data',
                        help='Directory name to save the model')
    parser.add_argument('--tmp_dir', type=str, default='../user_data/tmp_data', help='Directory name to save temp file')
    parser.add_argument('--result_dir', type=str, default='../prediction_result', help='Directory name to save result')
    parser.add_argument('--execution_num', type=int, default=1, help='The number of code executions')
    parser.add_argument('--process_model', action='store_false', help='The process model')
    parser.add_argument('--gpu_model', action='store_false', help='The GPU operation model')
    parser.add_argument('--gpu_option', type=int, default=0, help='The GPU id')
    parser.add_argument('--split_num', type=int, default=2, help='The number of hypotheses split')
    parser.add_argument('--retain_num', type=int, default=16, help='The maximal number of hypotheses to retain')
    parser.add_argument('--device_id', type=str, default='None', help='The device id')

    return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --epochs
    try:
        assert args.epochs >= 1
    except:
        print('The number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('The batch size must be larger than or equal to one')

    # --execution_num
    try:
        assert args.execution_num >= 1
    except:
        print('The execution number must be larger than or equal to one')

    # --split_num
    try:
        assert args.split_num >= 1
    except:
        print('The split number must be larger than or equal to one')

    # --retain_num
    try:
        assert args.split_num >= 1
    except:
        print('The retain number must be larger than or equal to one')

    return args


"""outputting config"""
def output_config(args):
    print('algorithm:' + args.algo_type)
    print('dataset:' + args.dataset)
    print('epochs:' + str(args.epochs))
    print('divide_type:' + str(args.divide_type))
    print('execution_num:' + str(args.execution_num))

    if args.algo_type == 'MLHE':
        print('split_num:' + str(args.split_num))
        print('retain_num:' + str(args.retain_num))


"""executing algorithm"""
def execute(ML, args):
    # declare instance for GAN
    if args.algo_type == 'restarttraining':
        result = ML.restarttraining()
    elif args.algo_type == 'retraining':
        result = ML.retraining()
    elif args.algo_type == 'MTsimple':
        result = ML.MTsimple()
    elif args.algo_type == 'MTht':
        result = ML.MTht()
    elif args.algo_type == 'MTcv':
        result = ML.MTcv()
    elif args.algo_type == 'MLHE':
        result = ML.MLHE(args.split_num, args.retain_num)
    else:
        raise Exception("[!] There is no option for " + args.algo_type)

    return result


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    output_config(args)

    results = []
    for i in range(args.execution_num):
        ML = MonotoneLearning(args)
        result = execute(ML, args)
        results.append(result)
    utils.output_results(results, args, i)

    print("Finished!")


if __name__ == '__main__':
    main()
