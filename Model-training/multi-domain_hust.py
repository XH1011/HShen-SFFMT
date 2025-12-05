import os
import sys
import torch
import numpy as np
import visdom
import time
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import normalized_mutual_info_score as nmi
from datetime import datetime
from sklearn.manifold import TSNE




import sys
sys.path.append('/hy-tmp')  # 手动加入 hy-tmp 的路径
# 获取项目根目录（论文4目录）的绝对路径
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# # 将根目录添加到sys.path
# sys.path.append(project_root)

from MTAGN_utils.train_utils import MyDataset, set_seed
from MTAGN_utils.plot_utils import plot_confusion_matrix, plot_fft
from data_generator.HUSTdata_domain import HUST_L1, HUST_L2, HUST_L3, HUST_L4
# from MTAGN_main.data_generator.SQdata import SQ_39, SQ_29, SQ_19
# from MTAGN_main.data_generator.EBdata import EB
from models.jlcnn import JLCNN
from models.mt1dcnn import MT1DCNN
from models.MTAGCN import MTAGCN
from models.No_ECA_attention import MTAGN_NoAtt
from torch.optim import RAdam #SGD
from sklearn.metrics import confusion_matrix
from models.mtagn_MixFeature import MTAGN
# from models.mtagn import MTAGN

TASK_TSNE_LABELS = {
    1: ['0', '1', '2', '3', '4'],   # task1 的类名（按你的类别索引顺序）
    2: ['0', '5', '6'],            # task2 的类名
}


# ==================== train and test =====================
class multi_task_trainer:
    def __init__(self, model_):
        self.model = model_

    def train(self, train_loader_, valid_loader_, valid_set_, optimizer_, scheduler_, epochs=100):
        # ----------- model initial ----------
        loss_fuc = torch.nn.CrossEntropyLoss()

        train_batch = len(train_loader_)
        valid_batch = len(valid_loader_)


        print('--------------------Training--------------------')
        counter = 1
        T = 2
        avg_cost = np.zeros([epochs, 6])  # 0\1\2 train loss; 3\4\5 valid loss
        lambda_weight = np.ones([2, epochs])
        weight = np.zeros((100, 2))
        for epoch in range(epochs):
            train_loss_1 = 0.0
            train_loss_2 = 0.0
            train_loss = 0.0
            train_acc_1 = 0
            train_acc_2 = 0
            true_label1_train = []
            true_label2_train = []
            pred_label1_train = []
            pred_label2_train = []


            cost = np.zeros(6, dtype=np.float32)

            if epoch == 0 or epoch == 1:
                lambda_weight[:, epoch] = 1.0
            else:
                w_1 = avg_cost[epoch - 1, 0] / avg_cost[epoch - 2, 0]
                w_2 = avg_cost[epoch - 1, 1] / avg_cost[epoch - 2, 1]
                lambda_weight[0, epoch] = 2 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))
                lambda_weight[1, epoch] = 2 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))

            self.model.train()
            for batch_idx, data in enumerate(train_loader_):
                optimizer_.zero_grad()
                inputs, label1, label2 = data
                inputs, label1, label2 = inputs.to(device), label1.to(device), label2.to(device)

                # forward + backward + update
                output1, output2 = self.model(inputs)
                tr_loss = [loss_fuc(output1, label1),
                           loss_fuc(output2, label2)]
                loss = sum(lambda_weight[i, epoch] * tr_loss[i] for i in range(2))

                train_loss_1 += tr_loss[0].item()
                train_loss_2 += tr_loss[1].item()
                train_loss += loss.item()

                loss.backward()
                optimizer_.step()

                cost[0] = tr_loss[0].item()
                cost[1] = tr_loss[1].item()
                cost[2] = loss.item()

                avg_cost[epoch, :3] += cost[:3] / train_batch

                train_acc_1 += np.sum(np.argmax(output1.cpu().data.numpy(), axis=1) == label1.cpu().numpy())
                train_acc_2 += np.sum(np.argmax(output2.cpu().data.numpy(), axis=1) == label2.cpu().numpy())

                pred_label1_batch = np.argmax(output1.cpu().data.numpy(), axis=1)
                pred_label2_batch = np.argmax(output2.cpu().data.numpy(), axis=1)

                pred_label1_train += pred_label1_batch.tolist()
                pred_label2_train += pred_label2_batch.tolist()
                true_label1_train += label1.cpu().numpy().tolist()
                true_label2_train += label2.cpu().numpy().tolist()
            train_acc_1 = (train_acc_1 / train_set.__len__()) * 100
            train_acc_2 = (train_acc_2 / train_set.__len__()) * 100
            train_F1_1 = f1_score(true_label1_train, pred_label1_train, average='macro')* 100
            train_F1_2 = f1_score(true_label2_train, pred_label2_train, average='macro')* 100
            train_NMI_1 = nmi(true_label1_train, pred_label1_train, average_method='geometric')* 100
            train_NMI_2 = nmi(true_label2_train, pred_label2_train, average_method='geometric')* 100


            # valid
            self.model.eval()
            with torch.no_grad():
                valid_loss_1 = 0.0
                valid_loss_2 = 0.0
                valid_loss = 0.0
                valid_acc_1 = 0
                valid_acc_2 = 0
                true_label1_valid = []
                true_label2_valid = []
                pred_label1_valid = []
                pred_label2_valid = []
                if epoch + 1 == epochs:
                    pred_label1 = np.array([], dtype=np.int64)
                    pred_label2 = np.array([], dtype=np.int64)
                for data in valid_loader_:
                    inputs, label1, label2 = data
                    inputs, label1, label2 = inputs.to(device), label1.to(device), label2.to(device)
                    output1, output2 = self.model(inputs)
                    val_loss = [loss_fuc(output1, label1),
                                loss_fuc(output2, label2)]
                    loss = sum(lambda_weight[i, epoch] * val_loss[i] for i in range(2))

                    valid_loss_1 += val_loss[0].item()
                    valid_loss_2 += val_loss[1].item()
                    valid_loss += loss.item()

                    cost[3] = val_loss[0].item()
                    cost[4] = val_loss[1].item()
                    cost[5] = loss.item()

                    avg_cost[epoch, 3:6] += cost[3:6] / valid_batch

                    valid_acc_1 += np.sum(np.argmax(output1.cpu().data.numpy(), axis=1) == label1.cpu().numpy())
                    valid_acc_2 += np.sum(np.argmax(output2.cpu().data.numpy(), axis=1) == label2.cpu().numpy())

                    pred_label1_batch = np.argmax(output1.cpu().data.numpy(), axis=1)
                    pred_label2_batch = np.argmax(output2.cpu().data.numpy(), axis=1)

                    pred_label1_valid += pred_label1_batch.tolist()
                    pred_label2_valid += pred_label2_batch.tolist()

                    true_label1_valid += label1.cpu().numpy().tolist()
                    true_label2_valid += label2.cpu().numpy().tolist()

                    if epoch + 1 == epochs:
                        pred_label1 = np.append(pred_label1, torch.max(output1, dim=1)[1].cpu().numpy().astype('int64'))
                        pred_label2 = np.append(pred_label2, torch.max(output2, dim=1)[1].cpu().numpy().astype('int64'))

                valid_acc_1 = 100 * valid_acc_1 / valid_set.__len__()
                valid_acc_2 = 100 * valid_acc_2 / valid_set.__len__()
                valid_F1_1 = f1_score(true_label1_valid, pred_label1_valid, average='macro')* 100
                valid_F1_2 = f1_score(true_label2_valid, pred_label2_valid, average='macro')* 100
                valid_NMI_1 = nmi(true_label1_valid, pred_label1_valid, average_method='geometric')* 100
                valid_NMI_2 = nmi(true_label2_valid, pred_label2_valid, average_method='geometric')* 100

                if epoch + 1 == epochs:
                    plot_confusion_matrix(valid_set_.labels1.numpy(), pred_label1, norm=True, task=1)
                    plot_confusion_matrix(valid_set_.labels2.numpy(), pred_label2, norm=True, task=2)
                    plt.show()
            # vis.line(Y=[[avg_cost[epoch, 0]*train_batch, avg_cost[epoch, 1]*train_batch, avg_cost[epoch, 2]*train_batch, avg_cost[epoch, 3]*train_batch,
            #              avg_cost[epoch, 4]*train_batch, avg_cost[epoch, 5]*train_batch]], X=[counter],
            #          update=None if counter == 0 else 'append', win='loss',
            #          opts=dict(legend=['train_loss_1', 'train_loss_2', 'train_loss', 'valid_loss_1', 'valid_loss_2',
            #                            'valid_loss'], title='Loss', ))
            vis.line(Y=[
                [avg_cost[epoch, 0] * train_batch, avg_cost[epoch, 1] * train_batch, avg_cost[epoch, 2] * train_batch]],
                X=[counter], update=None if counter == 0 else 'append', win='loss',
                opts=dict(legend=['train_loss_1', 'train_loss_2', 'train_loss', 'valid_loss_1', 'valid_loss_2',
                                  'valid_loss'], title='Loss', ))
            vis.line(Y=[[train_acc_1, train_acc_2, train_F1_1, train_F1_2, train_NMI_1, train_NMI_2, valid_acc_1, valid_acc_2, valid_F1_1, valid_F1_2, valid_NMI_1, valid_NMI_2]], X=[counter],
                     update=None if counter == 0 else 'append', win='metrics',
                     opts=dict(legend=['train_acc_1', 'train_acc_2', 'train_F1_1', 'train_F1_2', 'train_NMI_1', 'train_NMI_2', 'valid_acc_1', 'valid_acc_2', 'valid_F1_1', 'valid_F1_2', 'valid_NMI_1', 'valid_NMI_2'], title='Accuracy/F1/NMI'))
            vis.line(Y=[[lambda_weight[0, epoch], lambda_weight[1, epoch]]], X=[counter],
                     update=None if counter == 0 else 'append', win='weight',
                     opts=dict(legend=['weight1', 'weight2'], title='Weight'))
            counter += 1

            scheduler_.step()
            print(
                'epoch: [{}/{}] | Loss: {:.5f} | FTI_acc_tr: {:.2f}% | FSI_acc_tr: {:.2f}% | '
                'FTI_F1_tr: {:.2f}% | FSI_F1_tr: {:.2f}% | '
                'FTI_NMI_tr: {:.2f}% | FSI_NMI_tr: {:.2f}% | '
                'FTI_acc_val: {:.2f}% | FSI_acc_val: {:.2f}% | '
                'FTI_F1_val: {:.2f}% | FSI_F1_val: {:.2f}% | '
                'FTI_NMI_val: {:.2f}% | FSI_NMI_val: {:.2f}% | '
                'w1:{:.3f} w2:{:.3f}'.format(
                    epoch + 1, epochs, train_loss, train_acc_1, train_acc_2,
                    train_F1_1, train_F1_2,
                    train_NMI_1, train_NMI_2,
                    valid_acc_1, valid_acc_2,
                    valid_F1_1, valid_F1_2,
                    valid_NMI_1, valid_NMI_2,
                    lambda_weight[0, epoch], lambda_weight[1, epoch]))
            weight[epoch, 0], weight[epoch, 1] = lambda_weight[0, epoch], lambda_weight[1, epoch]
        print('Finish training!')
        loss_out = avg_cost[:, :3] * train_batch
        print(loss_out.shape)
        np.savetxt('EB-train_loss-DWA.csv', loss_out, delimiter=',')
        print(weight.shape)
        print(weight)
        np.savetxt('lambda-weight-DWA.csv', weight, delimiter=',')
        order_save = input('Save model?(Y/N): ')
        if order_save == 'Y' or order_save == 'y':
            self.save(filename=model_dir, model_name_pkl=model_name)

    def test(self, test_set_, test_loader_):
        self.load(model_path)
        self.model.eval()

        print('-------------------- Testing --------------------')
        acc_1 = 0
        acc_2 = 0
        pred_label1 = np.array([], dtype=np.int64)
        pred_label2 = np.array([], dtype=np.int64)
        with torch.no_grad():
            true_label1 = []
            true_label2 = []
            t0 = time.time()
            for data in test_loader_:
                inputs, label1, label2 = data
                inputs = inputs.to(device)
                output1, output2 = self.model(inputs)

                acc_1 += np.sum(torch.max(output1, dim=1)[1].cpu().numpy() == label1.numpy())
                acc_2 += np.sum(torch.max(output2, dim=1)[1].cpu().numpy() == label2.numpy())

                pred_label1 = np.append(pred_label1, torch.max(output1, dim=1)[1].cpu().numpy().astype('int64'))
                pred_label2 = np.append(pred_label2, torch.max(output2, dim=1)[1].cpu().numpy().astype('int64'))

                true_label1 += label1.numpy().tolist()
                true_label2 += label2.numpy().tolist()
                t1 = time.time()

            FTI_acc = 100 * acc_1 / test_set_.__len__()
            FSI_acc = 100 * acc_2 / test_set_.__len__()
            FTI_f1 = f1_score(true_label1, pred_label1, average='macro')* 100
            FSI_f1 = f1_score(true_label2, pred_label2, average='macro')* 100
            FTI_nmi = nmi(true_label1, pred_label1, average_method='geometric')* 100
            FSI_nmi = nmi(true_label2, pred_label2, average_method='geometric')* 100
            te_time = t1 - t0

            print('Accuracy and F1 on test_dataset:')
            print(f'FTI-task: {FTI_acc:.2f}%   [{acc_1}/{test_set_.__len__()}]')
            print(f'FSI-task: {FSI_acc:.2f}%   [{acc_2}/{test_set_.__len__()}]')
            print(f'FTI-task F1 Score: {FTI_f1:.2f}%')
            print(f'FSI-task F1 Score: {FSI_f1:.2f}%')
            print(f'FTI-task NMI Score: {FTI_nmi:.2f}%')
            print(f'FSI-task NMI Score: {FSI_nmi:.2f}%')
            print(f'Test time: {te_time:.4f}')
            # plot_confusion_matrix(test_set_.labels1.numpy(), pred_label1, norm=True, task=1)
            # plot_confusion_matrix(test_set_.labels2.numpy(), pred_label2, norm=True, task=2)
            plot_confusion_matrix(test_set_.labels1.numpy(), pred_label1, norm=True, task=1, fig_name="cm_task1_MTAGCN_HUST.png")
            plot_confusion_matrix(test_set_.labels2.numpy(), pred_label2, norm=True, task=2, fig_name="cm_task2_MTAGCN_HUST.png")


    def save(self, filename, model_name_pkl):
        if os.path.exists(filename):
            filename = os.path.join(filename, model_name_pkl)
        torch.save(self.model.state_dict(), filename)
        print(f'This model is saved at: {filename}')

    def load(self, filename):
        state = torch.load(filename)
        self.model.load_state_dict(state)
        print('Load model successfully from [%s]' % filename)

def _pick_logits_from_output(output, task: int):
    """
    从模型输出里挑选出当前 task 的 logits（尽量返回形状 [B, C] 的 Tensor）。
    兼容 Tensor / (list, tuple) / dict 三种情况。
    """

    # 1) 直接是张量
    if torch.is_tensor(output):
        return output

    # 2) 是 (list/tuple)：优先选第 task 个 2D 张量（[B, C]），否则取第一个张量
    if isinstance(output, (list, tuple)):
        two_d = [o for o in output if torch.is_tensor(o) and o.dim() == 2]
        if len(two_d) >= task:
            return two_d[task - 1]
        if two_d:
            return two_d[0]
        # 没有 2D，就找第一个张量
        for o in output:
            if torch.is_tensor(o):
                return o

    # 3) 是字典：按常见 key 试取；否则取第一个张量 value
    if isinstance(output, dict):
        preferred_keys = [f't{task}_logits', f'task{task}', f'out{task}', f'output{task}', 'logits', 'output']
        for k in preferred_keys:
            if k in output and torch.is_tensor(output[k]):
                return output[k]
        for v in output.values():
            if torch.is_tensor(v):
                return v

    raise TypeError("Cannot pick logits from model output; please check model.forward() return value.")

@torch.no_grad()
def collect_logits_and_labels(model, loader, device, task=1, max_points=None):
    """
    收集 logits 和标签；支持模型多输出；支持可选下采样（max_points）防止样本过多。
    """
    model.eval()
    feats, labs = [], []
    for data in loader:
        inputs, label1, label2 = data
        inputs = inputs.to(device)
        output = model(inputs)

        logits = _pick_logits_from_output(output, task).detach()
        if logits.dim() > 2:                      # 万一不是 [B, C]，就展平
            logits = torch.flatten(logits, 1)

        label = label1 if task == 1 else label2
        # label 既可能是 tensor 也可能是 numpy；统一成 numpy
        if torch.is_tensor(label):
            label_np = label.cpu().numpy()
        else:
            label_np = np.asarray(label)

        feats.append(logits.cpu().numpy())
        labs.append(label_np)

    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labs, axis=0)

    # 可选：随机下采样，避免 t-SNE 太慢
    if isinstance(max_points, int) and X.shape[0] > max_points:
        idx = np.random.RandomState(42).choice(X.shape[0], max_points, replace=False)
        X, y = X[idx], y[idx]

    return X, y

def plot_tsne_from_logits(model, loader, device, task=1, title=None,xlabel='First dimension', ylabel='Second dimention', save_path=None, legend_loc='upper right'):
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'Times New Roman'  # 全局字体设置
    X, y = collect_logits_and_labels(model, loader, device, task)
    n = X.shape[0]
    # perplexity 必须 < 样本数，给个稳健的自适应
    perp = int(min(30, max(5, n // 10)))
    tsne = TSNE(
        n_components=2, init='pca', learning_rate='auto',
        perplexity=perp, n_iter=1000, random_state=42, verbose=1
    )
    X2 = tsne.fit_transform(X)

    plt.figure(figsize=(7, 6), dpi=120)
    classes = np.unique(y)
    names = TASK_TSNE_LABELS.get(task)  # 仅用于图例

    # 为了图例顺序稳定，按类索引排序
    for c in sorted(classes.tolist()):
        idx = (y == c)
        label_name = (
            names[c] if (names is not None and isinstance(c, (int, np.integer)) and 0 <= c < len(names))
            else f'class {c}'
        )
        plt.scatter(X2[idx, 0], X2[idx, 1], s=12, alpha=0.7, label=label_name)
    # 坐标轴标题
    plt.xlabel(xlabel, fontsize=25, fontname='Times New Roman')
    plt.ylabel(ylabel, fontsize=25, fontname='Times New Roman')
    # ✅ 显示刻度
    plt.tick_params(axis='both', labelsize=25)
    # 确保刻度标签字体也是 Times New Roman
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontname('Times New Roman')
    legend = plt.legend(loc=legend_loc, markerscale=1.2, fontsize=25,
                        frameon=True, facecolor='white', framealpha=0.8)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(0.8)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print(f'Saved t-SNE figure to: {save_path}')
    plt.show()

if __name__ == '__main__':
    # ==================== Hyper parameters =====================
    EPOCHS = 100
    BATCH_SIZE = 50 #50
    LR = 0.0001#0.001
    # set_seed(2021)

    # define model, vis, optimiser
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    vis = visdom.Visdom(env='dwa')
    model = MTAGCN().to(device)
    # model = MTAGN().to(device)#MTAGN_NoAtt
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)#Adam RAdam
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

    # train_X, train_Y1, train_Y2, valid_X, valid_Y1, valid_Y2, test_X, test_Y1, test_Y2 = CW_1()
    train1_X, train1_Y1, train1_Y2, valid1_X, valid1_Y1, valid1_Y2= HUST_L1(is_source=True)
    train2_X, train2_Y1, train2_Y2, valid2_X, valid2_Y1, valid2_Y2 = HUST_L4(is_source=True)
    train3_X, train3_Y1, train3_Y2, valid3_X, valid3_Y1, valid3_Y2 = HUST_L2(is_source=True)

    # 拼接所有训练集数据
    train_X = np.concatenate([train1_X, train2_X, train3_X], axis=0)
    train_Y1 = np.concatenate([train1_Y1, train2_Y1, train3_Y1], axis=0)
    train_Y2 = np.concatenate([train1_Y2, train2_Y2, train3_Y2], axis=0)

    # 拼接所有y验证集数据
    valid_X = np.concatenate([valid1_X, valid2_X, valid3_X], axis=0)
    valid_Y1 = np.concatenate([valid1_Y1, valid2_Y1, valid3_Y1], axis=0)
    valid_Y2 = np.concatenate([valid1_Y2, valid2_Y2, valid3_Y2], axis=0)

    test_X, test_Y1, test_Y2 = HUST_L3(is_source=False)
    # ----------- train data ----------
    train_set = MyDataset(train_X, train_Y1, train_Y2)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
    print('train_set')
    print("train_X:", train_X.shape)
    print("train_X min:", train_X.min())
    print("train_X max:", train_X.max())
    print("train_X mean:", train_X.mean())
    print("train_X std:", train_X.std())
    print("train_Y1:", train_Y1.shape)
    print("train_Y2:", train_Y2.shape)
    print(f"任务1（label1）类别数: {len(np.unique(train_Y1))}, 类别分别是: {np.unique(train_Y1)}")
    print(f"任务2（label2）类别数: {len(np.unique(train_Y2))}, 类别分别是: {np.unique(train_Y2)}")

    # ----------- valid data ----------
    valid_set = MyDataset(valid_X, valid_Y1, valid_Y2)
    valid_loader = DataLoader(valid_set, shuffle=False, batch_size=BATCH_SIZE)
    print('valid_set')
    print("valid_X:", valid_X.shape)
    print("valid_Y1:", valid_Y1.shape)
    print("valid_Y2:", valid_Y2.shape)
    # ----------- data ----------
    test_set = MyDataset(test_X, test_Y1, test_Y2)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=BATCH_SIZE)
    print("test_X:", test_X.shape)
    print("test_Y1:", test_Y1.shape)
    print("test_Y2:", test_Y2.shape)

    print(f"验证集 - 任务1类别: {np.unique(valid_Y1)}")
    print(f"验证集 - 任务2类别: {np.unique(valid_Y2)}")

    print(f"测试集 - 任务1类别: {np.unique(test_Y1)}")
    print(f"测试集 - 任务2类别: {np.unique(test_Y2)}")

    trainer = multi_task_trainer(model)

    # 模型名字及保存地址
    model_dir = r'/hy-tmp/MTAGN'
    model_name = 'MTAGN_eca6_dwa-cw1-100-5.pkl'
    # model_name = 'test.pkl'
    model_path = os.path.join(model_dir, model_name)
    print("Model path: ", model_dir)
    if not os.path.exists(model_dir):
        print(f'Root dir {model_dir} does not exit.')
        exit()
    else:
        print('File exist? ', os.path.exists(model_dir))
    # order_tr = input('Train or not?(Y/N): ')
    # if order_tr == 'Y' or order_tr == 'y':
    trainer.train(train_loader, valid_loader, valid_set, optimizer, scheduler, EPOCHS)

    # order_te = input("Test or not?(Y/N): ")
    # if order_te == "Y" or order_te == "y":
    trainer.test(test_set, test_loader)

    # 第二步：只有做过测试，才会进入 t-SNE 的提问
    order_tsne = input("Plot t-SNE-2 from logits on VALID set? (Y/N): ")
    if order_tsne in ["Y", "y"]:
        plot_tsne_from_logits(model, test_loader, device, task=2,
                                xlabel='First Dimension', ylabel='Second Dimension',
                                save_path='/hy-tmp/tsne_MTAGCN_HUST_task2.png')

    # 第三步：无论 task2 选没选，都可以选择 task1
    order_tsne_test = input("Plot t-SNE-1 from logits on VALID set? (Y/N): ")
    if order_tsne_test in ["Y", "y"]:
        plot_tsne_from_logits(model, test_loader, device, task=1,
                                xlabel='First Dimension', ylabel='Second Dimension',
                                save_path='/hy-tmp/tsne_MTAGCN_HUST_task1.png')
