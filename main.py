import argparse
import os
import time
from tqdm import tqdm
import numpy as np
from torch import nn
from preprocessing.dataloader import MyDataLoader
import torch
import random
import torch.optim.lr_scheduler as sched
from sklearn.metrics import f1_score
from models.E2Former import E2foremr
from common.utils import check_class_distribution, check_feature_distribution

class EarlyStopping:
    def __init__(self, patience=5, delta=0, mode="min"):
        """
        patience: 允许验证集性能不提升的轮数
        delta: 认为性能提升的最小变化量
        mode: "min" 表示监控损失（越小越好），"max" 表示监控指标（越大越好）
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == "min" and score > self.best_score + self.delta) or \
             (self.mode == "max" and score < self.best_score - self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class Trainer():
    def __init__(self, net, optimizer, epochs,
                      use_cuda=True, gpu_num=0,
                      checkpoint_folder="./checkpoints",
                      max_lr=0.1, min_mom=0.7,
                      max_mom=0.99, num_classes=13,
                      sample_weights=None):

        self.optimizer = optimizer
        self.epochs = epochs
        self.use_cuda = use_cuda
        self.gpu_num = gpu_num
        self.checkpoints_folder = checkpoint_folder
        self.max_lr = max_lr
        self.min_mom = min_mom
        self.max_mom = max_mom
        self.num_classes = num_classes

        # 确保 sample_weights 的长度与类别数一致
        if sample_weights is not None and len(sample_weights) == self.num_classes:
            sample_weights = sample_weights.clone().detach()
        else:
            sample_weights = None
        self.criterion = nn.CrossEntropyLoss(weight=sample_weights)
        self.val_criterion = nn.CrossEntropyLoss()

        if self.use_cuda:
            if sample_weights is not None:
                self.criterion.weight = sample_weights.clone().detach().cuda('cuda:%i' %self.gpu_num)
            self.net = net.cuda('cuda:%i' % self.gpu_num)
        else:
            self.net = net

    def train(self, train_loader, eval_loader):
        # 初始化OneCycleLR学习率调度器
        scheduler = sched.OneCycleLR(self.optimizer, max_lr=self.max_lr, epochs=self.epochs,
                                     steps_per_epoch=len(train_loader),
                                     pct_start=0.425, anneal_strategy='linear', cycle_momentum=True,
                                     base_momentum=self.min_mom, max_momentum=self.max_mom,
                                     div_factor=10.0, three_phase=True, final_div_factor=10)

        best_f1 = 0
        best_loss = 0
        best_acc = 0

        early_stopping = EarlyStopping(patience=5, mode="max")  # 监控 F1 分数

        for epoch in range(self.epochs):
            """
               start：记录当前epoch的开始时间
               running_loss_train：累计训练损失
               running_loss_eval：累计验证损失
               train_total：训练样本总数
               train_correct：训练正确预测的样本数
               train_y_pred：训练集的预测标签
               train_y_true：训练集的真实标签
               total：验证样本总数
               correct：验证正确预测的样本数
               y_pred：验证集的预测标签
               y_true：验证集的真实标签
            """
            start = time.time()
            running_loss_train = 0.0
            running_loss_eval = 0.0
            train_total = 0.0
            train_correct = 0.0
            train_y_pred = torch.empty(0)
            train_y_true = torch.empty(0)
            total = 0.0
            correct = 0.0
            y_pred = torch.empty(0)
            y_true = torch.empty(0)

            self.net.train()

            for inputs, labels in tqdm(train_loader, total=len(train_loader), desc='Train round', unit='batch', leave=False):
                eye, eeg = inputs

                if self.use_cuda:
                    eye, eeg = eye.cuda('cuda:%i' % self.gpu_num).float(), eeg.cuda('cuda:%i' % self.gpu_num).float()
                    labels = labels.cuda('cuda:%i' % self.gpu_num).long()

                self.optimizer.zero_grad()

                outputs = self.net(eye, eeg)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()
                scheduler.step()

                # 累计当前 epoch 中所有批次的训练损失
                running_loss_train += loss.item()

                _, predicted = torch.max(outputs.data, 1)  # 沿着第 1 维度（即类别维度）找到最大值的索引

                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                train_y_pred = torch.cat((train_y_pred, predicted.view(predicted.shape[0]).cpu()))  # 保存当前 epoch 中所有批次的预测结果
                train_y_true = torch.cat((train_y_true, labels.view(labels.shape[0]).cpu()))   # 保存当前 epoch 中所有批次的真实标签

            end = time.time()

            train_acc = 100 * train_correct / train_total
            train_f1 = f1_score(train_y_true, train_y_pred, average='macro')

            self.net.eval()

            with torch.no_grad():
                for inputs, labels in tqdm(eval_loader, total=len(eval_loader), desc='Val round', unit='batch', leave=False):
                    eye, eeg = inputs

                    if self.use_cuda:
                        eye, eeg = eye.cuda('cuda:%i' % self.gpu_num).float(), eeg.cuda(
                            'cuda:%i' % self.gpu_num).float()
                        labels = labels.cuda('cuda:%i' % self.gpu_num).long()

                    eval_outputs = self.net(eye, eeg)
                    eval_loss = self.val_criterion(eval_outputs, labels)
                    running_loss_eval += eval_loss.item()  # 累计当前 epoch 中所有批次的测试损失

                    _, predicted = torch.max(eval_outputs.data, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    y_pred = torch.cat((y_pred, predicted.view(predicted.shape[0]).cpu()))
                    y_true = torch.cat((y_true, labels.view(labels.shape[0]).cpu()))

            acc = 100 * correct / total
            f1 = f1_score(y_true, y_pred, average='macro')

            print('Epoch {:03d}: Loss {:.4f}, Accuracy {:.4f}, F1 score {:.4f} || Val Loss {:.4f}, Val Accuracy {:.4f}, Val F1 score {:.4f}  [Time: {:.4f}]'
                .format(epoch + 1, running_loss_train / len(train_loader), train_acc, train_f1,
                        running_loss_eval / len(eval_loader), acc, f1, end - start))

            early_stopping(f1)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if f1 > best_f1:
                best_f1 = f1
                best_loss = running_loss_eval / len(eval_loader)
                best_acc = acc

                torch.save(self.net.state_dict(), self.checkpoints_folder + "/best_" + str(epoch+1) + ".pt")

                print("Save Best Checkpoints!  Best val loss: {:.4f}    Best val acc: {:.4f}    Best val f1: {:.4f}"
                      .format(best_loss, best_acc, best_f1))

        print(f'Finished Training')


class Tester():
    def __init__(self, net, use_cuda=True, gpu_num=0, checkpoint_path=None):
        self.use_cuda = use_cuda
        self.gpu_num = gpu_num
        self.checkpoint_path = checkpoint_path

        # 加载模型
        self.net = net
        if self.use_cuda:
            self.net = self.net.cuda('cuda:%i' % self.gpu_num)

        # 加载保存的模型权重
        if self.checkpoint_path:
            self.net.load_state_dict(torch.load(self.checkpoint_path))
            print(f"Loaded model from {self.checkpoint_path}")
        else:
            print("No checkpoint path provided. Testing with randomly initialized model.")

    def test(self, test_loader):
        self.net.eval()

        y_true = torch.empty(0)
        y_pred = torch.empty(0)
        total = 0
        correct = 0
        running_loss = 0.0

        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, total=len(test_loader), desc='Testing', unit='batch', leave=False):
                eye, eeg = inputs

                if self.use_cuda:
                    eye, eeg = eye.cuda('cuda:%i' % self.gpu_num).float(), eeg.cuda('cuda:%i' % self.gpu_num).float()
                    labels = labels.cuda('cuda:%i' % self.gpu_num).long()

                outputs = self.net(eye, eeg)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                y_pred = torch.cat((y_pred, predicted.view(predicted.shape[0]).cpu()))
                y_true = torch.cat((y_true, labels.view(labels.shape[0]).cpu()))

        acc = 100 * correct / total
        f1 = f1_score(y_true, y_pred, average='macro')
        avg_loss = running_loss / len(test_loader)

        print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {acc:.4f}, Test F1 Score: {f1:.4f}')

        return avg_loss, acc, f1

def main(args, n_workers):
    num_classes = args.num_classes
    lr = args.max_lr / 10

    train_loader, eval_loader, sample_weights = MyDataLoader(train_file=args.train_file_path,
                                                             test_file=args.test_file_path,
                                                             batch_size=args.batch_size,
                                                             num_workers=n_workers)

    # check_class_distribution(train_loader, dataset_name="Train Set")
    # check_class_distribution(eval_loader, dataset_name="Validation Set")
    # check_feature_distribution(train_loader, dataset_name="Train Set", feature_index=0)
    # check_feature_distribution(eval_loader, dataset_name="Validation Set", feature_index=0)

    net = E2foremr(eeg_channel=args.eeg_channel, eye_channel=args.eye_channel, num_classes=num_classes)

    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Number of parameters:', params)
    print()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.weight_decay, eps=1e-8)

    if args.train:
        trainer = Trainer(net, optimizer, epochs=args.epochs,
                          use_cuda=args.cuda, gpu_num=args.gpu_num,
                          checkpoint_folder=args.checkpoint_folder,
                          max_lr=args.max_lr, min_mom=args.min_mom,
                          max_mom=args.max_mom, num_classes=num_classes,
                          sample_weights=sample_weights)

        trainer.train(train_loader, eval_loader)

    if args.test:
        tester = Tester(net, use_cuda=args.cuda, gpu_num=args.gpu_num,
                          checkpoint_path=args.checkpoint_path)

        tester.test(eval_loader)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eeg_channel', type=int, default=32)
    parser.add_argument('--eye_channel', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--train_file_path', type=str, default='dataset/hci_datasets/train_augmented_data_Arsl.pt')
    parser.add_argument('--test_file_path', type=str, default='dataset/hci_datasets/test_data_Arsl.pt')
    parser.add_argument('--num_workers', default=1)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dropout_rate', type=int, default=0.2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--max_lr', type=float, default=0.00001)
    parser.add_argument('--min_mom', type=float, default=0.7)
    parser.add_argument('--max_mom', type=float, default=0.8)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--checkpoint_folder', type=str, default='checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/best/v1/best_24.pt')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    seed = args.seed
    n_workers = args.num_workers

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)

    main(args, n_workers)