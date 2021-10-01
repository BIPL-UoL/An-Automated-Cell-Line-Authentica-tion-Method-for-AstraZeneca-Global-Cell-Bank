from __future__ import print_function, division

import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image
import os,glob,itertools,tqdm
import random

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class CellRegression(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train_data_path,label_id_path_file=None,train=True, transform=None,cv_test_fold_id=1):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train_data_path = train_data_path
        self.train = train
        self.transform = transform

        self.class_number = 0
        self.cv_test_fold_id = cv_test_fold_id
        if label_id_path_file==None:
            self.label_id_path_file = './train__idx.txt'
        else:
            self.label_id_path_file = label_id_path_file
        self.load_data()

    def get_id_file(self):
        return self.label_id_path_file
    def get_file(self,path):

        ends = os.listdir(path)[0].split('.')[-1]

        img_list = glob.glob(os.path.join(path , '*.'+ends))
        return img_list

    def load_data(self):
        '''Measure one category'''

        categories = []

        for i in np.arange(1, 6, 1):
            if i != self.cv_test_fold_id:
                train_data_path = os.path.join(self.train_data_path, 'Fold{}'.format(i))
                print(train_data_path)
                categories.extend(list(map(self.get_file,
                                           list(map(lambda x: os.path.join(train_data_path, x),
                                                    os.listdir(train_data_path))))))

        # categories = list(map(self.get_file, list(map(lambda x: os.path.join(self.train_data_path, x), os.listdir(self.train_data_path)))))
        data_list = list(itertools.chain.from_iterable(categories))
        random.Random(10).shuffle(data_list)
        self.images_data ,self.labels_idx,self.labels= [],[],[]

        with_platform = os.name

        for file in tqdm.tqdm(data_list):


            img = Image.open(file)

            if self.transform is not None:
                img = self.transform(img)


            # Regression label from the file name
            if with_platform == 'posix':
                filename = file.split('/')[-1]


            elif with_platform=='nt':
                filename = file.split('\\')[-1]

            # remove jpg
            filename = filename.split('.')[0]
            # '00d12h00m'
            file_cTime = filename[-9:]
            #To hour-style ~
            label_time = (float(file_cTime[0:2]) * 24) + float(file_cTime[3:5]) + (float(file_cTime[6:8]) / 60)
            # To day-style ~
            #label_time = float(file_cTime[0:2]) + float(file_cTime[3:5])/24 + float(file_cTime[6:8]) /(60*24)
            # print('img:',file,' has label:',label)
            #img = img_to_array(img)
            self.images_data.append(img)

            self.labels_idx.append(label_time)

        self.max_timeP = max(self.labels_idx)



    def __len__(self):
        return len(self.labels_idx)

    def __getitem__(self, index):

        img, target = self.images_data[index], self.labels_idx[index]

        '''Without Normalize'''
        time_point = target
        #time_point = target / self.max_timeP




        return img, time_point
