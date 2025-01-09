"""
based on https://github.com/KevinJ-Huang/ExposureNorm-Compensation/blob/main/DRBN_ENC/create_txt.py
"""
import os
import os.path as osp
import argparse
from tqdm import tqdm

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def main():
    assert os.path.exists(inputdir), 'Input dir not found'
    assert os.path.exists(targetdir), 'target dir not found'
    mkdir(outputdir)
    imgs = sorted(os.listdir(inputdir))
    for idx,img in tqdm(enumerate(imgs)):
        groups = ''

        groups += os.path.join(inputdir, img) + '|'
        assert os.path.exists(os.path.join(inputdir, img)), f'{os.path.join(inputdir, img)} not found'
        # groups += os.path.join(args.auxi, img) + '|'
        gt_img = osp.join(targetdir, "_".join(osp.splitext(img)[0].split('_')[:-1])+'.JPG')
        groups += gt_img
        assert os.path.exists(gt_img), f'{gt_img} not found'
        # groups += os.path.join(targetdir,img)


        # if idx >= 800:
        #     break

        with open(os.path.join(outputdir, 'groups_test_mixexposure.txt'), 'a') as f:
            f.write(groups + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """
    python create_dataset_path.py --input 
    python create_dataset_path.py --target /root/workspace/RECNet/datasets/exposure_related/MSEC/training/GT_IMAGES --output . --input /root/workspace/RECNet/datasets/exposure_related/MSEC/multidir/training/INPUT/Exp1
    python create_dataset_path.py --target /root/workspace/RECNet/datasets/exposure_related/MSEC/validation/GT_IMAGES --output . --input /root/workspace/RECNet/datasets/exposure_related/MSEC/multidir/validation/INPUT/Exp1
    
    python create_dataset_path.py --target /root/workspace/RECNet/datasets/exposure_related/MSEC/testing/expert_a_testing_set --output . --input /root/workspace/RECNet/datasets/exposure_related/MSEC/multidir/testing/INPUT/Exp1
    """
    parser.add_argument('--input', type=str, default='/home/jieh/Dataset/Continous/Exposure/test/input/Exp5/', metavar='PATH', help='root dir to save low resolution images')
    # parser.add_argument('--auxi', type=str, default='/home/jieh/Dataset/Continous/ExpFiveLarge/train/Mid', metavar='PATH', help='root dir to save high resolution images')
    parser.add_argument('--target', type=str, default='/home/jieh/Dataset/Continous/Exposure/test/input/Exp3/', metavar='PATH', help='root dir to save high resolution images') 
    parser.add_argument('--output', type=str, default='/home/jieh/Projects/Continous/UEN_DRBN/data/', metavar='PATH', help='output dir to save group txt files')
    parser.add_argument('--ext', type=str, default='.png', help='Extension of files')
    args = parser.parse_args()

    inputdir = args.input
    targetdir = args.target
    outputdir = args.output
    ext = args.ext

    main()