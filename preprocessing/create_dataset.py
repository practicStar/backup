import argparse
import os
import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split
from scipy import signal
import copy

from tqdm import tqdm

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def list_files(directory, sorted_dir):
    files = []
    if sorted_dir:
        list_dir = sorted(os.listdir(directory), key=lambda filename: int(filename.split('.')[0]))
    else:
        list_dir = os.listdir(directory)
    for filename in list_dir:
        # if filename.endswith(".csv"):
        single = os.path.join(directory, filename)
        files.append(single)
    return files

class DatasetCreator:
    """
    preproc_data: 预处理数据的路径
    label_kind: 标签类型
    physio_f: 生理数据的采样频率
    gaze_f: 眼动数据的采样频率
    block_len: 从每个样本的末尾提取的秒数
    sample_len: 生成的每个样本的长度（秒）
    overlap: 样本之间的重叠比例
    verbose: 是否打印详细信息
    """
    def __init__(self, preproc_data, label_kind,
                 physio_f, gaze_f, block_len, sample_len, overlap, verbose=False):
        self.preproc_data = preproc_data
        self.label_kind = label_kind
        self.physio_f = physio_f
        self.gaze_f = gaze_f
        self.block_len = block_len
        self.sample_len = sample_len
        self.overlap = overlap
        self.verbose = verbose

    def save_to_list(self):
        sub_dir = list_files(self.preproc_data, sorted_dir=False)
        data_list = []
        labels = []
        for dir in tqdm(sub_dir, desc='Reading data'):
            subj = int(dir[-2:])

            all_labels = np.genfromtxt(os.path.join(dir, 'labels_felt{}.csv'.format(self.label_kind)), delimiter=',')

            # 获取所有实验的id
            id_trials = [x.split("\\")[-1].partition("_")[0] for x in list_files(dir, sorted_dir=False)]
            id_trials = sorted(np.unique(id_trials)[:-1], key=lambda x: int(x))
            for i, id in enumerate(tqdm(id_trials, desc=f'Subject {subj}')):
                pupil_data = np.genfromtxt(os.path.join(dir, '{}_PUPIL.csv'
                                    .format(id)), delimiter=',')
                gaze_data = np.genfromtxt(os.path.join(dir, '{}_GAZE_COORD.csv'
                                    .format(id)), delimiter=',')
                eye_dist_data = np.genfromtxt(os.path.join(dir, '{}_EYE_DIST.csv'
                                    .format(id)), delimiter=',')
                eeg_data = np.genfromtxt(os.path.join(dir, '{}_EEG.csv'
                                     .format(id)), delimiter=',')


                # 计算 眼动数据块/生理数据块 的总点数（采样点数）    数据块长度（秒） × 采样频率（Hz）
                n_points_block_gaze = self.block_len * self.gaze_f
                n_points_block_physio = self.block_len * self.physio_f

                pupil_data = pupil_data[-n_points_block_gaze:]
                gaze_data = gaze_data[-n_points_block_gaze:]
                eye_dist_data = eye_dist_data[-n_points_block_gaze:]
                eeg_data = eeg_data[-n_points_block_physio:]

                # 计算 眼动数据/生理数据 中每个样本的总点数（采样点数）   样本长度（秒） × 采样频率（Hz）
                n_points_sample_gaze = self.sample_len * self.gaze_f
                n_points_sample_physio = self.sample_len * self.physio_f

                # 计算 眼动数据/生理数据 中样本之间的重叠步长（点数）   样本点数 × 重叠比例
                overlap_step_gaze = int(n_points_sample_gaze * self.overlap)
                overlap_step_physio = int(n_points_sample_physio * self.overlap)

                for j, k in zip(range(0, n_points_block_gaze - overlap_step_gaze, n_points_sample_gaze - overlap_step_gaze),
                                range(0, n_points_block_physio - overlap_step_physio, n_points_sample_physio - overlap_step_physio)):
                    pupil = pupil_data[j : j + n_points_sample_gaze]
                    gaze_coord = gaze_data[j : j + n_points_sample_gaze]
                    eye_dist = eye_dist_data[j : j + n_points_sample_gaze]
                    eeg = eeg_data[k : k + n_points_sample_physio]


                    # 如果无效数据过多，则跳过当前样本
                    clean_pupil = pupil[pupil != -1]
                    clean_gaze_coord = gaze_coord[gaze_coord != -1]
                    clean_eye_dist = eye_dist[eye_dist != -1]
                    if len(clean_pupil)/len(pupil) < 0.6 or len(clean_gaze_coord)/len(gaze_coord) < 0.6 or len(clean_eye_dist)/len(eye_dist) < 0.6:
                        continue

                    eye = np.column_stack((pupil, gaze_coord, eye_dist))

                    eye = torch.FloatTensor(eye)
                    eeg = torch.FloatTensor(eeg)
                    label = all_labels[i]

                    if self.label_kind != "Emo":
                        if label <=3: label = 0
                        elif label >=7: label = 2
                        else: label = 1

                    data_list.append([eye, eeg])
                    labels.append(label)

        return data_list, labels

def std_for_SNR(signal, noise, snr):
    # 计算信号的方差，作为信号的功率
    signal_power = np.var(signal.numpy())
    # 计算噪声的方差，作为噪声的功率
    noise_power = np.var(noise.numpy())

    g = np.sqrt(10.0 ** (-snr/10) * signal_power / noise_power)
    return g

def add_gauss_noise(sample, snr):
    sample = copy.deepcopy(sample)
    sample_size = sample[1].size()[0]
    gaussian_noise = torch.empty(sample_size).normal_(mean=0, std=1)

    ch_noise = torch.FloatTensor(signal.resample(gaussian_noise.numpy(), sample[0].size()[0]))
    for i in range(4):
        idx = np.where(sample[0][:, i] != -1)[0]
        # 计算噪声的标准差，以达到指定的信噪比 snr
        sample[0][idx, i] = sample[0][idx, i] + std_for_SNR(sample[0][idx, i], ch_noise, snr) * ch_noise[idx].T

    for i in range(1, len(sample)):
        if len(sample[i].size()) > 1:
            noise = gaussian_noise.repeat(sample[i].size()[1],1).t()
        else:
            noise = gaussian_noise
        sample[i] = sample[i] + std_for_SNR(sample[i], noise, snr) * noise
    return sample

def scaled_sample(sample, min, max):
    sample = copy.deepcopy(sample)
    alpha = random.uniform(min, max)

    # 对眼动数据缩放
    for i in range(4):
        idx = np.where(sample[0][:, i]!=-1)[0]
        sample[0][idx, i] = alpha * sample[0][idx, i]

    # 对生理数据缩放
    for i in range(1, len(sample)):
        sample[i] = alpha * sample[i]
    return sample

def load_dataset(data, labels, scaling, noise, m, SNR):
    dataset = []

    for i in range(len(data)):
        sample = data[i]
        label= labels[i]
        dataset.append([sample, label])
        if scaling:
            # 对原始样本进行缩放增强
            dataset.append([scaled_sample(sample, 0.7, 0.8), label])
            dataset.append([scaled_sample(sample, 1.2, 1.3), label])
            if noise:
                # 对原始样本和缩放后的样本添加噪声  确保每个样本生成 m 个增强样本
                for j in range(0, m-3, 3):
                    dataset.append([add_gauss_noise(sample, snr=SNR), label])
                    dataset.append([add_gauss_noise(scaled_sample(sample, 0.7, 0.8), snr=SNR), label])
                    dataset.append([add_gauss_noise(scaled_sample(sample, 1.2, 1.3), snr=SNR), label])
        elif noise:
            for j in range(m-1):
                dataset.append([add_gauss_noise(sample, snr=SNR), label])

    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preproc_data_path', type=str, default='../dataset/hci_preproc_data')
    parser.add_argument('--save_path', type=str, default='../dataset/hci_datasets')
    parser.add_argument('--label_kind', type=str, default='Arsl', help="(Emo, Vlnc, Arsl) label")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', type=bool, action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    assert args.label_kind in ["Emo", "Arsl", "Vlnc"]
    print("Creating dataset for label: ", args.label_kind)

    set_seed(args.seed)

    d = DatasetCreator(args.preproc_data_path, args.label_kind, physio_f = 128, gaze_f = 60, block_len = 30, sample_len=10, overlap = 0, verbose=args.verbose)  # create object
    data, labels = d.save_to_list()
    print(len(data))

    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=args.seed, stratify=labels)

    m = 30
    SNR = 5

    train_data = load_dataset(X_train, y_train, scaling=True, noise=True, m=m, SNR=SNR)
    test_data = load_dataset(X_test, y_test, scaling=False, noise=False, m=1, SNR=None)

    print("Len train before augmentation: ", len(X_train))
    print("Len train after augmentation: ", len(train_data))
    print("Len test: ", len(test_data))
    print("Tot dataset: ", len(train_data) + len(test_data))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    torch.save(train_data, f'{args.save_path}/train_augmented_data_{args.label_kind}.pt')
    torch.save(test_data,  f'{args.save_path}/test_data_{args.label_kind}.pt')
