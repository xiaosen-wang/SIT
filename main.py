import os
import argparse
import tqdm
from utils import *
from attack import *

def get_parser():
    parser = argparse.ArgumentParser(description='Generating transferable adversaria examples')
    parser.add_argument('--eval', action='store_true', help='attack/evluation')
    parser.add_argument('--epoch', default=10, type=int, help='the iterations for updating the adversarial patch')
    parser.add_argument('--batchsize', default=64, type=int, help='the bacth size')
    parser.add_argument('--eval_batchsize', default=8, type=int, help='the bacth size')
    parser.add_argument('--eps', default=16/255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--alpha', default=1.6/255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--num_copies', default=20, type=int, help='number of copies')
    parser.add_argument('--num_block', default=3, type=int, help='number of shuffled blocks')
    parser.add_argument('--momentum', default=0., type=float, help='the decay factor for momentum based attack')
    parser.add_argument('--model', default='resnet18', type=str, help='the source surrogate model', 
                        choices=['resnet18', 'resnet101', 'resnext50','densenet121', 'mobilenet', 'vit', 'swin','inceptionv3'])
    parser.add_argument('--input_dir', default='data/', type=str, help='the path for the benign images')
    parser.add_argument('--output_dir', default='./results', type=str, help='the path to store the adversarial patches')
    parser.add_argument('--output_txt',default='trans.txt',type=str,help='the path to store the results')
    return parser.parse_args()

def main():
    args = get_parser()
    f2l = load_labels(os.path.join(args.input_dir, 'val_rs.csv'))
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not args.eval:
        model = wrap_model(model_list[args.model](weights='DEFAULT').eval().cuda())
        attacker = SIA(model, args.eps, args.alpha, args.epoch, args.momentum,args.num_copies,args.num_block)
        for batch_idx, [filenames, images] in tqdm.tqdm(enumerate(load_images(os.path.join(args.input_dir,'data'), args.batchsize))):
            labels = get_labels(filenames, f2l)
            perturbations = attacker(images, labels)
            save_images(args.output_dir, images + perturbations.cpu(), filenames)
        accuracy = dict()
        res = '|'
        for model_name, model_arc in model_list.items():
            model = wrap_model(model_list[args.model](weights='DEFAULT').eval().cuda())
            succ, total = 0, 0
            for batch_idx, [filenames, images] in tqdm.tqdm(enumerate(load_images(args.output_dir, args.eval_batchsize))):
                labels = get_labels(filenames, f2l)
                pred = model(images.cuda())
                succ += (labels.numpy() == pred.argmax(dim=1).detach().cpu().numpy()).sum()
                total += labels.shape[0]
            accuracy[model_name] = (succ / total * 100)
            print(model_name, accuracy[model_name])
            res += ' {:.2f} |'.format(accuracy[model_name])
        print(accuracy)
        print(res)
        with open(args.output_txt,'a+') as f:
            f.write(res)
    else:
        accuracy = dict()
        res = '|'
        for model_name, model_arc in model_list.items():
            if type(model_arc) is int:
                    continue
            model = wrap_model(model_arc(weights='DEFAULT').eval().cuda())
            succ, total = 0, 0
            for batch_idx, [filenames, images] in tqdm.tqdm(enumerate(load_images(args.output_dir, args.batchsize))):
                labels = get_labels(filenames, f2l)
                pred = model(images.cuda())
                succ += (labels.numpy() == pred.argmax(dim=1).detach().cpu().numpy()).sum()
                total += labels.shape[0]
            accuracy[model_name] = (succ / total * 100)
            print(model_name, accuracy[model_name])
            res += ' {:.2f} |'.format(accuracy[model_name])
        print(accuracy)
        print(res)
        with open(args.output_txt,'a+') as f:
            f.write(res)
            f.write('\r\n')

            
if __name__ == '__main__':
    main()
    