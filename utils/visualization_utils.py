import matplotlib.pyplot as plt
from matplotlib.text import Text
import pandas as pd
import seaborn as sns
import torch
from torchvision.utils import make_grid
from utils import scannet_utils


def visualize_confusion_matrix(writer, conf_matrix, ious, data_loader, epoch):
    df_cm = pd.DataFrame(conf_matrix.value(normalized=True), index=range(data_loader.num_classes),
                         columns=range(data_loader.num_classes))
    class_names = ['{}: {}'.format(i, x) for i, x in enumerate(data_loader.class_names)]
    df_cm.index, df_cm.columns = class_names, class_names
    df_cm["IoU"] = ious
    df_cm["Total"] = conf_matrix.value(normalized=False).sum(axis=1)
    df_cm = df_cm.drop(class_names[data_loader.ignore_classes], axis=0)
    df_cm = df_cm.drop(class_names[data_loader.ignore_classes], axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral', vmin=0.0, vmax=1.0, fmt='.2f', cbar=False).get_figure()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.xticks(rotation=90)
    for text in ax.findobj(Text):
        x, y = text.get_position()
        if len(df_cm.columns) - 1 <= x < len(df_cm.columns):
            text.set_text('{}'.format(df_cm["Total"][int(y)]))
            text.set_fontsize(int(text.get_fontsize() / 2) + 1)

    #plt.show()
    plt.close(fig_)
    writer.add_figure("Confusion matrix", fig_, epoch)


def visualize_tensor(writer, name, data, single=True, normalize=False, range=(-1.0, 1.0)):
    """format and display data on tensorboard"""
    disp = data[0, :, :, :].unsqueeze(0) if single else data
    writer.add_image(name, make_grid(disp, nrow=8, normalize=normalize, range=range))


def visualize_labels(writer, name, data, single=True, normalize=False):
    """format and display data on tensorboard"""
    lb, lh, lw = data.shape

    if single:
        lb = 1
        disp = data[0:1, :, :]
    else:
        disp = data

    with torch.no_grad():
        label_color = torch.zeros((lb, lh, lw, 3))
        color_palette = torch.from_numpy(scannet_utils.valid_color_palette() / 255.0).float()
        for idx, color in enumerate(color_palette):
            label_color[disp == idx] = color
        label_color = label_color.permute(0, 3, 1, 2)

    writer.add_image(name, make_grid(label_color, nrow=8, normalize=normalize))
