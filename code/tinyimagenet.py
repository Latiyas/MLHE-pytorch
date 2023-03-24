import os, sys, pickle, time, torch, random, utils
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch import FloatTensor, div
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, KFold
from timm.data.mixup import Mixup
from timm.data.random_erasing import RandomErasing
from math import ceil
from timm.loss.cross_entropy import SoftTargetCrossEntropy
from timm.models import create_model
from tqdm.auto import tqdm
from timm.utils.clip_grad import dispatch_clip_grad
from torchvision.io import read_image
import cv2 as cv


########## load TinyImageNet data set ##########
def load_TinyImageNet(root):
    root_dir = root
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")

    images_train = create_class_idx_dict_train(train_dir)
    images_val = create_class_idx_dict_val(val_dir)

    return images_train, images_val


def create_class_idx_dict_train(train_dir):
    if sys.version_info >= (3, 5):
        classes = [d.name for d in os.scandir(train_dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    classes = sorted(classes)
    class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    images = []
    img_root_dir = train_dir
    list_of_dirs = [target for target in class_to_tgt_idx.keys()]

    for tgt in list_of_dirs:
        dirs = os.path.join(img_root_dir, tgt)
        if not os.path.isdir(dirs):
            continue

        for root, _, files in sorted(os.walk(dirs)):
            for fname in sorted(files):
                if (fname.endswith(".JPEG")):
                    path = os.path.join(root, fname)
                    item = (path, class_to_tgt_idx[tgt])
                    images.append(item)

    return images


def create_class_idx_dict_val(val_dir):
    val_annotations_file = os.path.join(val_dir, "val_annotations.txt")
    val_img_to_class = {}
    set_of_classes = set()
    with open(val_annotations_file, 'r') as fo:
        entry = fo.readlines()
        for data in entry:
            words = data.split("\t")
            val_img_to_class[words[0]] = words[1]
            set_of_classes.add(words[1])

    classes = sorted(list(set_of_classes))
    class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    images = []
    img_root_dir = val_dir
    list_of_dirs = ["images"]

    for tgt in list_of_dirs:
        dirs = os.path.join(img_root_dir, tgt)
        if not os.path.isdir(dirs):
            continue

        for root, _, files in sorted(os.walk(dirs)):
            for fname in sorted(files):
                if (fname.endswith(".JPEG")):
                    path = os.path.join(root, fname)
                    item = (path, class_to_tgt_idx[val_img_to_class[fname]])
                    images.append(item)

    return images


class TinyImageNet(Dataset):
    def __init__(self, dataset, train=True, transform=None, normalize=None):
        self.Train = train
        self.dataset = dataset
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, tgt = self.dataset[idx]
        with open(img_path, 'rb') as f:
            data = read_image(img_path)
            if data.shape[0] == 1:
                data = torch.tensor(cv.cvtColor(data.permute(1, 2, 0).numpy(), cv.COLOR_GRAY2RGB)).permute(2, 0, 1)

        if self.transform:
            data = self.transform(data)

        data = div(data.type(FloatTensor), 255)
        if self.normalize:
            data = self.normalize(data)

        return data, tgt


class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer,
                 clip_grad=None, clip_mode='norm', parameters=None,
                 create_graph=False, update_grad=False
                 ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if clip_grad:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
        if update_grad:
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_data_augmentations(args, en_mixup=True, en_cutmix=True, en_randerase=True):
    mixup = None
    random_erase = None
    m_alpha = 0.8 if en_mixup else 0
    c_alpha = 1.0 if en_cutmix else 0
    if en_mixup or en_cutmix:
        mixup = Mixup(
            mixup_alpha=m_alpha,
            cutmix_alpha=c_alpha,
            label_smoothing=0.1,
            num_classes=200
        )
    if en_randerase:
        if args.gpu_model:
            random_erase = RandomErasing(
                probability=0.25,
                mode='pixel',
                device=args.gpu_option
            )
        else:
            random_erase = RandomErasing(
                probability=0.25,
                mode='pixel'
            )

    return mixup, random_erase


# Restart training the model each iteration on TinyImageNet
class tinyimagenet_utils:
    def __init__(self, args):
        # parameter
        self.args = args
        self.name = 'TinyImageNet'
        self.train_times = 1
        self.batch_num = 32
        self.worker_num = 0
        self.learning_rate = 1e-3
        self.channel_size = 3
        self.picture_size = 64
        # output status: 0-save data, 1-save picture
        self.o_state = self.args.output_model
        # data path
        self.data_path = '../data/{}/tiny-imagenet-200/'.format(self.name)
        # temp path
        self.tmp_path = self.args.tmp_dir + '/{}'.format(self.name)
        # result path
        self.result_path = self.args.result_dir
        # configure
        self.divide_type = self.args.divide_type
        self.epoch_num = self.args.epochs
        # size of training subset and test subset
        self.train_num = int(100000 / self.epoch_num)
        self.test_num = int(10000 / self.epoch_num)

        self.trainsets, self.testsets = load_TinyImageNet(self.data_path)

        # get index of sampling
        index_state = utils.getIndex(self.divide_type)

        train_path = self.tmp_path + '/{}_state{}_train({}).pkl'.format(self.name, index_state, self.epoch_num)
        test_path = self.tmp_path + '/{}_state{}_test({}).pkl'.format(self.name, index_state, self.epoch_num)
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            self.generate_sample_index(index_state)

        self.train_slice = utils.load_data(train_path)
        self.test_slice = utils.load_data(test_path)

        self.mixup, self.random_erase = get_data_augmentations(self.args, en_mixup=True, en_cutmix=True,
                                                               en_randerase=True)

        label_smooth = 0.1
        if self.mixup:
            self.loss_fn = SoftTargetCrossEntropy()
        else:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    # get the hyperparameters of the base model
    def get_paras(self):
        return self.epoch_num, self.train_times, self.learning_rate

    # get base model
    def get_model(self):
        model = create_model('swin_large_patch4_window12_384', pretrained=True, drop_path_rate=0.1)

        for param in model.parameters():
            param.requires_grad = False
        model.reset_classifier(num_classes=200)

        return model

    # split--Multithreading
    def mul_scatter(self, o_models, scatter_num, dataloader):
        # number of threads
        thread_num = 1

        # copy
        tmp = []
        if not o_models:
            clf = self.get_model()
            for j in range(scatter_num):
                tmp.append(deepcopy(clf))
        else:
            for o_model in o_models:
                for j in range(scatter_num):
                    tmp.append(deepcopy(o_model))

        # train
        model_num = len(tmp)
        n_models = []
        for i in range(model_num):
            n_model = self.train_model(tmp[i], dataloader)
            n_models.append(deepcopy(n_model))

        return n_models

    # train model
    def train_model(self, model, train_loader):
        if self.args.gpu_model:
            model = model.to(self.args.gpu_option)

        true_batch_size = 128
        update_freq = true_batch_size // self.batch_num
        loss_scaler = NativeScaler()

        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=ceil(len(train_loader) / update_freq) * self.train_times)

        if not self.args.process_model:
            iterator = tqdm(train_loader, total=int(len(train_loader)))
        else:
            iterator = train_loader

        update = False

        # deit finetunes in eval mode
        if type(model).__name__ == 'VisionTransformerDistilled':
            model.eval()
        else:
            model.train()

        for epoch in range(self.train_times):
            for i, (x, y) in enumerate(iterator):
                if self.args.gpu_model:
                    x, y = x.to(self.args.gpu_option, non_blocking=True), y.to(self.args.gpu_option, non_blocking=True)

                if self.mixup:
                    x, y = self.mixup(x, y)

                if self.random_erase:
                    x = self.random_erase(x)

                update = True if (i + 1) % update_freq == 0 or i + 1 == len(train_loader) else False
                with torch.cuda.amp.autocast():
                    pred = model(x)
                    loss = self.loss_fn(pred, y)
                loss_scaler(loss, optimizer, update_grad=update)

                if update:
                    for param in model.parameters():
                        param.grad = None
                    scheduler.step()

        model.cpu()

        return model

    # test model
    def test_model(self, model, test_loader):
        if self.args.gpu_model:
            model = model.to(self.args.gpu_option)

        if not self.args.process_model:
            iterator = tqdm(test_loader, total=int(len(test_loader)))
        else:
            iterator = test_loader

        # count the number of samples that are correctly predicted
        correct_num = 0
        sample_num = 0
        # define contingency table
        contingency = []

        model.eval()
        with torch.no_grad():
            for _, (x, y) in enumerate(iterator):
                if self.args.gpu_model:
                    x, y = x.to(self.args.gpu_option), y.to(self.args.gpu_option)
                outputs = model(x)

                _, predicted = torch.max(outputs, 1)

                for j in range(len(predicted)):
                    sample_num += 1
                    predicted_num = predicted[j].item()
                    label_num = y[j].item()
                    # compare y and ^y
                    if predicted_num == label_num:
                        correct_num += 1
                        contingency.append(True)
                    else:
                        contingency.append(False)

            # calculation top-1 accuracy
            correct_rate = correct_num / sample_num

        model.cpu()

        return correct_rate, contingency

    # generate sample index
    def generate_sample_index(self, state=0):
        # the sample index of training set and test set
        train_slice = []
        test_slice = []

        # sequential segmentation
        if state == 1:
            # raw training set is read by category
            train_arr = list(range(self.epoch_num * self.train_num))
            # set seed
            random.seed(3)
            random.shuffle(train_arr)

            test_arr = list(range(self.epoch_num * self.test_num))

            for i in range(self.epoch_num):
                train_slice.append(train_arr[i * self.train_num: (i + 1) * self.train_num])
                test_slice.append(test_arr[i * self.test_num: (i + 1) * self.test_num])
        # random sampling (put back)
        elif state == 2:
            for i in range(self.epoch_num):
                train_slice.append(random.sample(range(self.train_num * self.epoch_num), self.train_num))
                test_slice.append(random.sample(range(self.test_num * self.epoch_num), self.test_num))
        # stratified sampling (not put back)
        elif state == 3:
            skf = StratifiedKFold(n_splits=self.epoch_num, shuffle=True, random_state=5)
            train_y = []
            for key in self.trainsets:
                train_y.append(key[1])

            test_y = []
            for key in self.testsets:
                test_y.append(key[1])

            for _, train_index in skf.split(self.trainsets, train_y):
                train_slice.append(train_index.tolist())

            for _, test_index in skf.split(self.testsets, test_y):
                test_slice.append(test_index.tolist())
        else:
            raise Exception("[!] There is no option for " + state)

        # save index as .pkl file
        with open(self.tmp_path + '/{}_state{}_train({}).pkl'.format(self.name, state, self.epoch_num), 'wb') as f:
            pickle.dump(train_slice, f)
        with open(self.tmp_path + '/{}_state{}_test({}).pkl'.format(self.name, state, self.epoch_num), 'wb') as f:
            pickle.dump(test_slice, f)

    # get subsets by index
    def get_Isubset(self, train_slice, test_slice):
        # get training subset
        train = [self.trainsets[i] for i in train_slice]
        n_train = train
        # get test subset
        test = [self.testsets[i] for i in test_slice]
        n_test = test

        return n_train, n_test

    # get data sets under different strategies
    def get_dataset(self, i):
        # replaceing old data;
        if self.divide_type in [1, 3, 5]:
            # get index based on different sampling
            train_slice = self.train_slice[i]
            test_slice = self.test_slice[i]
            n_train, n_test = self.get_Isubset(train_slice, test_slice)

            self.train = n_train
            self.test = n_test
        # replaying old data;
        elif self.divide_type in [2, 4, 6]:
            # get index based on different sampling
            train_slice = self.train_slice[i]
            test_slice = self.test_slice[i]
            n_train, n_test = self.get_Isubset(train_slice, test_slice)

            if i == 0:
                self.train = n_train
                self.test = n_test
            else:
                self.train = self.train + n_train
                self.test = self.test + n_test

        return self.train, self.test

    # get dataloader: s_state-decide whether to shuffle the dataset;
    def get_dataloader(self, data, train=False):
        if train:
            transform = transforms.Compose([
                transforms.Resize(384, interpolation=InterpolationMode.BICUBIC),
                transforms.RandAugment(num_ops=2, magnitude=9),
            ])

            datasets = TinyImageNet(data, train=True, transform=transform, normalize=transforms.Compose(
                [transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ]))

            dataloader = DataLoader(dataset=datasets, shuffle=True, batch_size=self.batch_num,
                                    num_workers=self.worker_num, drop_last=True)
        else:
            transform = transforms.Compose([
                transforms.Resize(384, interpolation=InterpolationMode.BICUBIC),
            ])

            datasets = TinyImageNet(data, train=False, transform=transform, normalize=transforms.Compose(
                [transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ]))

            dataloader = DataLoader(dataset=datasets, shuffle=False, batch_size=self.batch_num,
                                    num_workers=self.worker_num)

        return dataloader

    # get the index of k-fold
    def get_k_fold(self, k):
        return utils.get_k_fold(self.train_num, self.test_num, k)

    # divide data according to k1
    def get_k_fold_data(self, k1, trainsets, testsets, train_index, test_index):
        train = []
        test = []
        for i in range(len(train_index)):
            if train_index[i] != k1:
                train.append(trainsets[i])
            else:
                test.append(trainsets[i])

        for i in range(len(test_index)):
            if test_index[i] != k1:
                train.append(testsets[i])
            else:
                test.append(testsets[i])

        n_train = train
        n_test = test

        return n_train, n_test
