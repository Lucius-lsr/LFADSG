from config import parse_args
from model import LFADSG
import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from scipy.signal import savgol_filter
import matplotlib.patches as mpatches
import torch.nn as nn


MODEL_PATH = '/home/zhangzizhao/tmp/pycharm_project_919/log/models_2020-10-16-20-43-37/best_model.pt'
EVAL_PATH = '/home/zhangzizhao/tmp/pycharm_project_919/dataset/allen_data/eval_data/10-9'
COMPARE_MODEL1_PATH = '/home/zhangzizhao/tmp/pycharm_project_919/log/models_2020-10-16-20-43-37/best_model.pt'
COMPARE_MODEL2_PATH = '/home/zhangzizhao/tmp/pycharm_project_919/log/models_2020-10-16-20-50-20/best_model.pt'
COMPARE_MODEL3_PATH = '/home/zhangzizhao/tmp/pycharm_project_919/log/models_2020-10-16-21-16-46/best_model.pt'
COMPARE_MODEL4_PATH = '/home/zhangzizhao/tmp/pycharm_project_919/log/models_2020-10-16-21-23-24/best_model.pt'

PEARSON1_PATH = '/home/zhangzizhao/tmp/pycharm_project_919/dataset/neuropixel_data/pearson_gabors_VISp.npy'
PEARSON2_PATH = '/home/zhangzizhao/tmp/pycharm_project_919/dataset/neuropixel_data/pearson_natural_movie_one_more_repeats_VISp.npy'
PEARSON3_PATH = '/home/zhangzizhao/tmp/pycharm_project_919/dataset/neuropixel_data/pearson_drifting_gratings_75_repeats_VISp.npy'
PEARSON4_PATH = '/home/zhangzizhao/tmp/pycharm_project_919/dataset/neuropixel_data/pearson_dot_motion_VISp.npy'


def get_model(model_path):
    arg = parse_args()
    model = LFADSG(time=arg.time, neurons=arg.neurons, dim_e=arg.dim_e, dim_c=arg.dim_c, \
        dim_d=arg.dim_d, gcn_hidden=arg.gcn_hidden, tau=arg.tau, time_lag=arg.time_lag, \
            dropout_prob=arg.dropout_prob, kl_weight_1=arg.kl_weight_1, kl_weight_2=arg.kl_weight_2, \
                autocorrelation_taus=arg.autocorrelation_taus, noise_variances=arg.noise_variances, \
                    dynamic_graph=arg.dynamic_graph, discrete_graph=arg.discrete_graph)

    print('load model from', model_path)
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model


def get_eval_data(data_path):
    data_file = os.listdir(data_path)

    fre_ori_data = []
    for file in data_file:
        path = os.path.join(data_path, file)
        data = np.load(path)
        frequency, oritation = file[:-4].split('_')
        fre_ori_data.append((frequency, oritation, data))

    return fre_ori_data
    

def draw_curves(pddata, class_column, value_column, ax, draw_legend=False):
    classes = pddata[class_column].to_numpy()
    color_list = list(mcolors.TABLEAU_COLORS.items())
    patches = []

    for i in range(pddata.shape[0]):
        color = color_list[i][1]
        curve_group = pddata.iloc[i][value_column]
        for curve in curve_group:
            smoothed_curve = savgol_filter(curve, 31, 3)
            ax.plot(smoothed_curve, linewidth = '0.4', color=color)
        patches.append(mpatches.Patch(color=color, label='{}: {}'.format(class_column, classes[i])))
    if draw_legend:
        ax.legend(handles=patches)


def consistency_eval():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(MODEL_PATH)
    fre_ori_data = get_eval_data(EVAL_PATH)
    condition_list = list(zip(*[t[0:2] for t in fre_ori_data]))
    # fre_list = np.array(condition_list[0])
    # ori_list = np.array(condition_list[1])
    rates_pd=pd.DataFrame(columns={"fre","ori","neuron","trialxtime"})
    raw_pd=pd.DataFrame(columns={"fre","ori","neuron","trialxtime"})
    # fres, fres_index = np.unique(fre_list, return_inverse=True)
    # oris, oris_index = np.unique(ori_list, return_inverse=True)
    # rates = np.empty_like(np.dstack([t[2] for t in fre_ori_data]))

    # painter = CurvePainter()
    for fre, ori, raw_data in fre_ori_data:
        data = torch.from_numpy(raw_data).to(device)

        with torch.no_grad():
            rate_list=[]
            for single_data in data:
                print(single_data.shape)
                single_data = single_data.unsqueeze(0)
                rate = model(single_data, return_rates=True)
                rate_list.append(rate)
            rates = torch.cat(rate_list, dim=0)
            print(rates.shape)
            # rates = data
        for neuron in range(rates.shape[2]):
            rates_pd.loc[rates_pd.shape[0]]={"fre":fre,"ori":ori,"neuron":neuron,"trialxtime":rates[:,:,neuron].tolist()}
            raw_pd.loc[raw_pd.shape[0]]={"fre":fre,"ori":ori,"neuron":neuron,"trialxtime":raw_data[:,:,neuron].tolist()}
            # list_curves = []
            # for trial in range((rates.shape[0])):
            #     curve = rates[trial, :, neurons].tolist()
            #     painter.add_curve(fre, ori, neurons, curve)
                # list_curves.append(curve)
            # painter.draw_single(fre, ori, neurons, list_curves)
    
    fre_list = np.unique(rates_pd['fre'].to_numpy())
    ori_lis = np.unique(rates_pd['ori'].to_numpy())
    neuron_list = np.unique(rates_pd['neuron'].to_numpy())
    for neuron in neuron_list:
        rates_pd_filter = rates_pd.loc[(rates_pd["fre"] == fre_list[0]) & (rates_pd["neuron"] == neuron)]
        raw_pd_filter = raw_pd.loc[(raw_pd["fre"] == fre_list[0]) & (raw_pd["neuron"] == neuron)]
        fig = plt.figure()
        fig.tight_layout()
        ax1 = fig.add_subplot(121)
        draw_curves(rates_pd_filter, class_column='ori', value_column='trialxtime',ax=ax1)
        ax1.set_title("Inferred Rates")
        ax2 = fig.add_subplot(122)
        draw_curves(raw_pd_filter, class_column='ori', value_column='trialxtime',ax=ax2, draw_legend=True)
        ax2.set_title("Raw dF/F")
        plt.subplots_adjust(wspace =0.25, hspace =0)#调整子图间距
        file_path = '/home/zhangzizhao/tmp/pycharm_project_919/result/10-12/{}'.format(fre_list[0])
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        plt.savefig(file_path+'/{}.jpg'.format(neuron), format='jpg', dpi=300, pad_inches=0)
        plt.close()

    # for k1 in painter.curves_dict:
    #     for k2 in painter.curves_dict[k1]:
    #         for k3 in painter.curves_dict[k1][k2]:
    #             print(k1, k2, k3, len(painter.curves_dict[k1][k2][k3]))


class CurvePainter():
    def __init__(self):
        self.curves_dict = {}
    
    def add_curve(self, fre, ori, neurons, curve):
        dic_neurons = self.curves_dict.setdefault(neurons, {})
        dic_neurons_fre = dic_neurons.setdefault(fre, {})
        list_curves = dic_neurons_fre.setdefault(ori, [])
        list_curves.append(curve)

    def draw(self):
        pass
        # fontsize = 25
        # for k1 in painter.curves_dict:
        #     for k2 in painter.curves_dict[k1]:
        #         for k3 in painter.curves_dict[k1][k2]:
        #             list_curves = painter.curves_dict[k1][k2][k3]
        #             for curve in list_curves:
        #                 plt.plot(curve, linewidth = '1', color='red')
        #             plt.grid(linestyle='-.', linewidth=1)
        #             ax = plt.gca()
        #             ax.spines['bottom'].set_linewidth(2)
        #             ax.spines['left'].set_linewidth(2)
        #             ax.spines['right'].set_linewidth(2)
        #             ax.spines['top'].set_linewidth(2)
        #             plt.savefig('/home/zhangzizhao/tmp/pycharm_project_919/result/10-9/{}_{}_{}.jpg'.format(k1,k2,k3), format='jpg', dpi=300, pad_inches=0)
        #             plt.close()

    def draw_single(self, fre, ori, neurons, list_curves):
        for curve in list_curves:
            plt.plot(curve, linewidth = '1', color='yellow')
        mean_curve = np.array(list_curves)
        mean_curve = np.mean(mean_curve, axis=0)
        plt.plot(mean_curve, linewidth = '3', color='red')

        plt.grid(linestyle='-.', linewidth=1)
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        file_path = '/home/zhangzizhao/tmp/pycharm_project_919/result/10-9/{}/{}'.format(fre, ori)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        plt.savefig(file_path+'/{}.jpg'.format(neurons), format='jpg', dpi=300, pad_inches=0)
        plt.close()


def draw_adajency_matrix():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(MODEL_PATH)
    adajency_matrix = model.get_prior_graph()
    plt.matshow(adajency_matrix.detach().cpu().squeeze().numpy())
    plt.colorbar()
    file_path = '/home/zhangzizhao/tmp/pycharm_project_919/result/10-16'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    plt.savefig(file_path+'/adj_prob.jpg', format='jpg', dpi=300, pad_inches=0)
    plt.close()

def compare_latent_graph():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model1 = get_model(COMPARE_MODEL1_PATH)
    graph_prior_1 = model1.get_prior_graph().squeeze()
    model2 = get_model(COMPARE_MODEL2_PATH)
    graph_prior_2 = model2.get_prior_graph().squeeze()
    pearson1 = torch.sigmoid(torch.from_numpy(np.abs(np.load(PEARSON1_PATH))))
    pearson2 = torch.sigmoid(torch.from_numpy(np.abs(np.load(PEARSON2_PATH))))
    x = np.arange(0, 300, 1)
    y = np.zeros_like(x,dtype=np.float)
    criterion = nn.MSELoss()
    for i in range(len(x)):
        tmp1 = torch.sigmoid(torch.randn(graph_prior_1.size()))
        tmp2 = torch.sigmoid(torch.randn(graph_prior_2.size()))
        mseloss = criterion(tmp1,tmp2)
        # print(mseloss.numpy())
        y[i] = mseloss.numpy()
    plt.scatter(x, y, alpha=0.6)
    plt.axhline(y=np.mean(y), color='r', linestyle='-')
    print(criterion(graph_prior_1.float(),graph_prior_2.float()))
    plt.axhline(y=criterion(graph_prior_1.float(),graph_prior_2.float()), color='b', linestyle='-')
    plt.axhline(y=criterion(pearson1,pearson2), color='g', linestyle='-')
    file_path = '/home/zhangzizhao/tmp/pycharm_project_919/result/10-16'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    plt.savefig(file_path+'/adj_prob_comparison.jpg', format='jpg', dpi=300, pad_inches=0)
    plt.close()

def random_graph_from_prob(prob):
    graph = np.zeros_like(prob,dtype=float)
    for i in range(len(prob)):
        for j in range(len(prob[i])):
            graph[i][j] = np.random.choice([0,1],p=[1-prob[i][j],prob[i][j]])
    return graph

def compare_latent_graph_by_sampling():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model1 = get_model(COMPARE_MODEL1_PATH)
    graph_prior_1 = model1.get_prior_graph()
    graph_prior_1 = graph_prior_1.cpu().detach().squeeze().numpy()
    model2 = get_model(COMPARE_MODEL2_PATH)
    graph_prior_2 = model2.get_prior_graph()
    graph_prior_2 = graph_prior_2.cpu().detach().squeeze().numpy()

    pearson1 = np.abs(np.load(PEARSON1_PATH))
    pearson2 = np.abs(np.load(PEARSON2_PATH))
    x = np.arange(0, 300, 1)
    y1 = np.zeros_like(x,dtype=np.float)
    y2 = np.zeros_like(x,dtype=np.float)
    y3 = np.zeros_like(x,dtype=np.float)
    criterion = nn.MSELoss()
    for i in range(len(x)):
        tmp1 = torch.from_numpy(np.random.randint(0,2,size=graph_prior_1.size))
        tmp2 = torch.from_numpy(np.random.randint(0,2,size=graph_prior_2.size))
        mseloss1 = criterion(tmp1.float(),tmp2.float())
        print(mseloss1.numpy())
        y1[i] = mseloss1.numpy()
        tmp3 = torch.from_numpy(random_graph_from_prob(graph_prior_1))
        tmp4 = torch.from_numpy(random_graph_from_prob(graph_prior_1))
        mseloss2 = criterion(tmp3.float(),tmp4.float())
        print(mseloss2.numpy())
        y2[i] = mseloss2.numpy()
        tmp5 = torch.from_numpy(random_graph_from_prob(pearson1))
        tmp6 = torch.from_numpy(random_graph_from_prob(pearson1))
        mseloss3 = criterion(tmp5.float(),tmp6.float())
        print(mseloss3.numpy())
        y3[i] = mseloss3.numpy()

    plt.scatter(x, y1,  color='r', alpha=0.6)
    plt.axhline(y=np.mean(y1), color='r', linestyle='--')
    plt.scatter(x, y2,  color='b', alpha=0.6)
    plt.axhline(y=np.mean(y2), color='b', linestyle='--')
    plt.scatter(x, y3,  color='g', alpha=0.6)
    plt.axhline(y=np.mean(y3), color='g', linestyle='--')
    
    file_path = '/home/zhangzizhao/tmp/pycharm_project_919/result/10-16'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    plt.savefig(file_path+'/adj_sampling_comparison.jpg', format='jpg', dpi=300, pad_inches=0)
    plt.close()


# draw_adajency_matrix()
# consistency_eval()
compare_latent_graph()