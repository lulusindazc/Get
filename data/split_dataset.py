import os
import numpy as np
import torch
import torchvision
import random
import os.path as osp
import argparse

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



def make_dataset_with_labels(dir, classnames):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    images = []
    labels = []

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            dirname = os.path.split(root)[-1]
            if dirname not in classnames:
                continue
            label = classnames.index(dirname)
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                labels.append(label)

    return images, labels



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain Adaptation Methods: The ensemble version NA+noise')
    parser.add_argument('--method', type=str, default='uda', choices=['uda', 'pda', 'ssda'])
    parser.add_argument('--dset', type=str, default='OfficeHome',
                        choices=['DomainNet126', 'VisDA_2017', 'office31', 'OfficeHome'],
                        help="The dataset or source dataset used")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=2, help="target")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--partial', type=bool, default=False)
    parser.add_argument('--output', type=str, default='logs/uda/run_noise_debug/')
    parser.add_argument('--shot', type=int, default=1, choices=[1, 3])
    args = parser.parse_args()
    args.output = args.output.strip()
    args.data_root = '/media/zhangc/Data/Dataset/Domain_adaptation'  # '/ssd/czhang/data/da' /media/zhangc/Data/Dataset/Domain_adaptation #/ssd/czhang/data/da'#'/home/zhangc/data/DA_dataset # /temp/zhangc/data/da

    if args.dset == 'OfficeHome':
        names = ['Art', 'Clipart', 'Product', 'Real']
        args.class_num = 65
        if args.partial:
            args.tgt_class_name = 25
        # args.max_epoch=50
    if args.dset == 'office31':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
        # args.max_epoch=50
    if args.dset == 'DomainNet126':
        names = ['clipart', 'painting', 'real', 'sketch']
        args.class_num = 126
    if args.dset == 'VisDA_2017':
        names = ['train', 'validation']
        args.class_num = 12
        args.warmup_epoch = 1
        args.max_epoch = 1
    if not args.partial:
        args.tgt_class_name = args.class_num
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # args.s_dset_path = os.path.join(args.data_root, args.dset,
    #                                 names[args.s])  # './data/' + args.dset + '/' + names[args.s] + '_list.txt'
    # args.t_dset_path = os.path.join(args.data_root, args.dset,
    #                                 names[args.t])  # './data/' + args.dset + '/' + names[args.t] + '_list.txt'
    args.t_dset_path ='/home/zhangc/file/projects/pycharm_files/SSL/CaGenCFReg/'+ 'data/ssda/' + args.dset + '/unlabeled_target_images_' \
                        + 'Real_' + str(args.shot) + '.txt'
    args.t_dset_patht = '/home/zhangc/file/projects/pycharm_files/SSL/CaGenCFReg/'+'data/ssda/' + args.dset + '/unlabeled_target_images_' \
                       + 'RealWorld_' + str(args.shot) + '.txt'
    write_text=''
    labeltxt = open(args.t_dset_path)
    for line in labeltxt:
        data = line.strip().split(' ')
        # if is_image_file(data[0]):
        #     path = os.path.join(root, data[0])
        gt = int(data[1])
        write_text+=data[0].replace('Real','RealWorld')+' '+ str(gt)+'\n'

    with open(args.t_dset_patht,'w') as OFile:
        OFile.write(write_text)
    # if not osp.exists(args.output_dir):
    #     os.system('mkdir -p ' + args.output_dir)
    # if not osp.exists(args.output_dir):
    #     os.mkdir(args.output_dir)
    # with open(os.path.join(args.data_root, args.dset, 'category.txt'), 'r') as f:
    #     classes = f.readlines()
    #     args.classnames = [c.strip() for c in classes]
    #
    # if args.partial:
    #     args.classnames_tgt =args.classnames[:25]
    # else:
    #     args.classnames_tgt =args.classnames
    # args.test_dset_path = args.t_dset_path
    #
    # args.output_dir = osp.join(args.output, args.method, args.dset,
    #                            args.name)
    # args.log = osp.join(args.output_dir, "{:}_{:}_{:}.txt".format(args.method,args.dset,names[args.t]))
    # # if os.path.exists(trained_model_path):
    # #     args.out_file = open(osp.join(args.output_dir, "{:}.txt".format(args.log)), "a")
    # # else:
    # args.out_file = open(osp.join(args.output_dir, "{:}.txt".format(args.log)), "w")
    #
    # make_dataset_with_labels(args.t_dset_path, args.classnames)
