import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import AutoregressiveModel
from data import EarthquakeDataset

def test_model(model:AutoregressiveModel, dataloader:torch.utils.data.DataLoader):
    gt_list = []
    pred_list = []
    for iter, batch in enumerate(dataloader):
        sequence, next_value = batch
        gt_list.append(next_value)
        with torch.no_grad():
            pred = model(sequence)[:,0,0]
            pred_list.append(pred)
    gt_list = torch.cat(gt_list)
    pred_list = torch.cat(pred_list)
    l1_dist = torch.nn.functional.pairwise_distance(gt_list, pred_list,p=1)
    l2_dist = torch.nn.functional.pairwise_distance(gt_list, pred_list,p=2)
    return l1_dist, l2_dist, gt_list, pred_list

def main():
    model_path = "/Users/furkancoskun/Desktop/doktora_dersler/MMI-711/hw1/training_output/ckpt/epoch-199/pytorch_model.bin"
    test_out_path = "/Users/furkancoskun/Desktop/doktora_dersler/MMI-711/hw1/training_output/test_results/epoch-199"
    dataset_path = "/Users/furkancoskun/Desktop/doktora_dersler/MMI-711/hw1/hw data"
    test_txts = ["20170724065925_4823.txt"]
    window_length = 25
    hidden_size = 15
    num_layers = 3

    os.makedirs(test_out_path, exist_ok=True)
    model = AutoregressiveModel(input_dim=window_length,
                                hidden_size=hidden_size,
                                num_layers=num_layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    dataset = EarthquakeDataset(dataset_path, test_txts, window_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    l1_dist, l2_dist, gt_list, pred_list = test_model(model, dataloader)
    print(f"l1_dist: {l1_dist}")
    print(f"l2_dist: {l2_dist}")
    gt_list_np = gt_list.cpu().detach().numpy()
    pred_list_np = pred_list.cpu().detach().numpy()
    t = np.arange(0, gt_list_np.shape[0], 1, dtype=int)

    plot_window_size = 500
    for i in range(gt_list_np.shape[0]//plot_window_size):
        t_ = t[i*plot_window_size:(i+1)*plot_window_size]
        gt_list_np_ = gt_list_np[i*plot_window_size:(i+1)*plot_window_size]
        pred_list_np_ = pred_list_np[i*plot_window_size:(i+1)*plot_window_size]
        plt.figure()
        plt.plot(t_, gt_list_np_, 'r', t_, pred_list_np_, 'b')
        plt.legend(["ground-truth","prediction"], loc='upper left')
        plt.savefig(os.path.join(test_out_path, f"{i}.png"))
    t_ = t[(i+1)*plot_window_size:-1]
    gt_list_np_ = gt_list_np[(i+1)*plot_window_size:-1]
    pred_list_np_ = pred_list_np[(i+1)*plot_window_size:-1]
    plt.figure()
    plt.plot(t_, gt_list_np_, 'r', t_, pred_list_np_, 'b')
    plt.legend(["ground-truth","prediction"], loc='upper left')
    plt.savefig(os.path.join(test_out_path, f"{i+1}.png"))

if __name__ == "__main__":
    main()