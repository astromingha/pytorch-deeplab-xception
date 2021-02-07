import os
import shutil
import torch
from collections import OrderedDict
import glob
from dataloaders.datasets import radish, crops
import json
import pandas as pd
from mypath import Path

class Saver(object):

    def __init__(self, args):
        self.args = args
        # self.directory = os.path.join('run', args.dataset, args.checkname)
        # root_dir = os.path.split(os.path.split(Path.db_root_dir(args.dataset))[0])[0]
        root_dir = os.path.split(Path.db_root_dir(args.dataset))[0]
        self.directory = os.path.join(root_dir, 'results', args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        # if not os.path.exists(self.experiment_dir):
        #     os.makedirs(self.experiment_dir)

        # self.cityscapes_train = radish.LandcoverSegmentation(args, split='train')
        # self.cityscapes_train = crops.CropSegmentation(args, split='train')

    def save_validation_results(self,iou,confusion_matrix):
        dataframe1 = pd.DataFrame(iou)
        dataframe2 = pd.DataFrame(confusion_matrix)
        dataframe1.to_excel(excel_writer=os.path.join(self.directory, "iou.xlsx"))
        dataframe2.to_excel(excel_writer=os.path.join(self.directory, "confumatrix.xlsx"))
        # raw_data = {'col0': [1, 2, 3, 4],
        #            >> > 'col1': [10, 20, 30, 40],
        # >> > 'col2': [100, 200, 300, 400]}  # 리스트 자료형으로 생성
        # >> > raw_data = pd.DataFrame(raw_data)  # 데이터 프레임으로 전환
        # >> > raw_data.to_excel(excel_writer='sample.xlsx')  # 엑셀로 저장

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        self.cityscapes_train = crops.CropSegmentation(self.args, split='train')

        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['dataset'] = self.args.dataset
        p['backbone'] = self.args.backbone
        p['valid_class_num'] = len(self.cityscapes_train.valid_classes)
        p['mean'] = json.dumps(self.cityscapes_train.mean)
        p['std'] = json.dumps(self.cityscapes_train.std)
        p['out_stride'] = self.args.out_stride
        p['batch_size'] = self.args.batch_size
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size
        p['gpu_num'] = self.args.gpu_ids
        p['resume'] = self.args.resume
        p['use_balanced_weights'] = self.args.use_balanced_weights
        p['workers'] = self.args.workers

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()