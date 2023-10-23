import os
import argparse
from models.SFNet import build_net
from eval import _eval
from train import _train
# from train_ots import _train_ots
def get_parser():
    parser = argparse.ArgumentParser()

    # Directorise
    parser.add_argument('--model_name', type=str, default='SFNet')
    parser.add_argument('--gpu_num', type=str, default='1', choices=['0', '1'])
    # ../dataset/
    # dataset/
    # /home/jiangmao/code/dataset/
    parser.add_argument('--data_dir', type=str, default='')  # 到train的上层文件夹
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])

    # Train
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=1)
    parser.add_argument('--resume', type=str, default='')#恢复模型训练的路径
    parser.add_argument('--model_save_dir', type=str, default='result/Training-Results/')

    # Test
    parser.add_argument('--test_model', type=str, default='results/Training-Results/Best.pkl')
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])
    parser.add_argument('--result_dir', type=str, default='results/images/')

    args = parser.parse_args()


    return args

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num

    import torch
    from torch.backends import cudnn

    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    # if not os.path.exists('results/'+args.model_name+'/'):
    #     os.makedirs('results/'+args.model_name+'/')
    if not os.path.exists(args.result_dir):
        os.makedirs((args.result_dir))
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    mode = args.mode
    model = build_net(mode)

    #如果cuda可用，将模型移动到gpu上面进行计算
    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)
    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    args = get_parser()
    main(args)
