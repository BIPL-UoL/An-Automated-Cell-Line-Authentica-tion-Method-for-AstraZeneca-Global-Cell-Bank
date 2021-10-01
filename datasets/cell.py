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

class CellDataset_CV(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_path,label_id_path_file=None,train=True, transform=None,cv_test_fold_id=1,shuffle=True,Selected_cells=30):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.train = train
        self.transform = transform

        self.class_number = 0
        if label_id_path_file==None:
            self.label_id_path_file = './train_class_idx.txt'
        else:
            self.label_id_path_file = label_id_path_file

        self.cv_test_fold_id = cv_test_fold_id
        self.data_path = data_path
        self.shuffle = shuffle
        self.Selected_cells = Selected_cells
        self.load_data()

    def get_id_file(self):
        return self.label_id_path_file
    def get_file(self,path):

        ends = os.listdir(path)[0].split('.')[-1]

        img_list = glob.glob(os.path.join(path , '*.'+ends))
        return img_list

    def load_data(self):
        categories = []

        for i in np.arange(1,6,1):
            if i != self.cv_test_fold_id:

                train_data_path = os.path.join(self.data_path,'Fold{}'.format(i))
                print(train_data_path)
                all_cell_lines=os.listdir(train_data_path)[-self.Selected_cells:]

                categories.extend(list(map(self.get_file,
                                           list(map(lambda x: os.path.join(train_data_path, x),
                                                                   all_cell_lines)))))

        #categories = list(map(self.get_file, list(map(lambda x: os.path.join(self.train_data_path, x), os.listdir(self.train_data_path)))))
        data_list = list(itertools.chain.from_iterable(categories))
        if self.shuffle:
            random.Random(10).shuffle(data_list)
        else:
            pass
        self.images_data ,self.labels_idx,self.labels= [],[],[]

        with_platform = os.name

        for file in tqdm.tqdm(data_list):


            img = Image.open(file)

            if self.transform is not None:
                img = self.transform(img)



            if with_platform == 'posix':
                label = file.split('/')[-2]
            elif with_platform=='nt':
                label = file.split('\\')[-2]

            # print('img:',file,' has label:',label)
            #img = img_to_array(img)
            self.images_data.append(img)
            self.labels.append(label)


        if self.train:

            with open(self.label_id_path_file, 'w') as f:
                labels_strings = np.unique(self.labels)
                f.writelines([line + '\n' for line in labels_strings])

            with open(self.label_id_path_file,'r') as f:
                lines = f.readlines()
                lines = [line.rstrip() for line in lines]
                for label in self.labels:
                    idx = lines.index(label.rstrip())
                    self.labels_idx.append(idx)

            print(f"\tTraining data includes {np.unique(self.labels_idx).shape[0]} classes"
                  )
            #labels = np.to_categorical(np.array(labels_idx),num_classes=self.classNumber)
        else:
            #self.images_data = np.array(self.images_data, dtype='float32') / 255.0
            with open(self.label_id_path_file,'r') as f:
                lines = f.readlines()
                lines = [line.rstrip() for line in lines]
                for label in self.labels:
                    if label.rstrip() in lines:

                        idx = lines.index(label.rstrip())
                        self.labels_idx.append(idx)
                    # else:
                    #     unknown_label = len(lines)
                    #     self.labels_idx.append(unknown_label)

            # print(f"\tTesting data includes {np.unique(self.labels_idx).shape[0] } classes (Original {np.unique(self.labels).shape[0]} classes),"
            #       f" {len(self.labels_idx)} samples.")
        self.class_number = np.unique(self.labels_idx).shape[0]
    def __len__(self):
        return len(self.labels_idx)

    def __getitem__(self, index):

        img, target = self.images_data[index], self.labels_idx[index]



        return img, target





class CellDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train_data_path,label_id_path_file=None,train=True, transform=None):
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
        if label_id_path_file==None:
            self.label_id_path_file = './train_class_idx.txt'
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

        categories = list(map(self.get_file, list(map(lambda x: os.path.join(self.train_data_path, x), os.listdir(self.train_data_path)))))
        data_list = list(itertools.chain.from_iterable(categories))
        random.Random(10).shuffle(data_list)
        self.images_data ,self.labels_idx,self.labels= [],[],[]

        with_platform = os.name

        for file in tqdm.tqdm(data_list):


            img = Image.open(file)

            if self.transform is not None:
                img = self.transform(img)



            if with_platform == 'posix':
                label = file.split('/')[-2]
            elif with_platform=='nt':
                label = file.split('\\')[-2]

            # print('img:',file,' has label:',label)
            #img = img_to_array(img)
            self.images_data.append(img)
            self.labels.append(label)


        if self.train:

            with open(self.label_id_path_file, 'w') as f:
                labels_strings = np.unique(self.labels)
                f.writelines([line + '\n' for line in labels_strings])

            with open(self.label_id_path_file,'r') as f:
                lines = f.readlines()
                lines = [line.rstrip() for line in lines]
                for label in self.labels:
                    idx = lines.index(label.rstrip())
                    self.labels_idx.append(idx)

            print(f"\tTraining data includes {np.unique(self.labels_idx).shape[0]} classes"
                  )
            #labels = np.to_categorical(np.array(labels_idx),num_classes=self.classNumber)
        else:
            #self.images_data = np.array(self.images_data, dtype='float32') / 255.0
            with open(self.label_id_path_file,'r') as f:
                lines = f.readlines()
                lines = [line.rstrip() for line in lines]
                for label in self.labels:
                    if label.rstrip() in lines:

                        idx = lines.index(label.rstrip())
                        self.labels_idx.append(idx)
                    # else:
                    #     unknown_label = len(lines)
                    #     self.labels_idx.append(unknown_label)

            # print(f"\tTesting data includes {np.unique(self.labels_idx).shape[0] } classes (Original {np.unique(self.labels).shape[0]} classes),"
            #       f" {len(self.labels_idx)} samples.")
        self.class_number = np.unique(self.labels_idx).shape[0]
    def __len__(self):
        return len(self.labels_idx)

    def __getitem__(self, index):

        img, target = self.images_data[index], self.labels_idx[index]



        return img, target


