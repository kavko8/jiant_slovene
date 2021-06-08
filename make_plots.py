import matplotlib.pyplot as plt
import os
import jsonlines
from matplotlib.ticker import MaxNLocator
import numpy as np


def movingaverage(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def create(tasks, model_name, path_to_look, num_epochs, epoch_length):
    for path, _, files in os.walk(path_to_look):
        if "graph_steps.zlog" in files:
            graph_results_path = os.path.join(path, "graph_steps.zlog")
            with jsonlines.open(graph_results_path) as reader:
                scores = []
                scores_per_task = {}
                for task in tasks:
                    scores_per_task[task] = []
                for obj in reader:
                    obj = dict(obj)
                    score = float("%.2f" % (obj["score"] * 100))
                    scores.append(score)
                    for task in tasks:
                        task_score = float("%.2f" % (obj["metrics"][task]["major"] * 100))
                        scores_per_task[task].append(task_score)

        if "loss_train.zlog" in files:
            loss_resuts_path = os.path.join(path, "loss_train.zlog")
            with jsonlines.open(loss_resuts_path) as reader:
                loss = {}
                avg_loss_per_epochs = {}
                for task in tasks:
                    loss[task] = []
                    avg_loss_per_epochs[task] = []
                for obj in reader:
                    task = obj["task"]
                    c_loss = obj["loss_val"]
                    loss[task].append(c_loss)
                for task in tasks:
                    epoch_list = []
                    values = loss[task].copy()

                    for j, i in enumerate(values):
                        if ((j + 1) % epoch_length) == 0:
                            suma = sum(values[j - epoch_length + 1:j + 1]) / epoch_length
                            epoch_list.append(suma)
                    avg_loss_per_epochs[task] = epoch_list

    y1 = scores
    num_epochs = len(scores)
    x = [i + 1 for i in range(num_epochs)]
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel(f'Epoch ({epoch_length} steps in 1 epoch)')
    ax1.set_ylabel('Accuracy (major) (%)', color=color)
    x_ax = ax1.axes.get_xaxis()
    x_ax.set_major_locator(MaxNLocator(integer=True))
    ax1.plot(x, y1, color=color, marker="o")
    ax1.set_xticks(np.arange(start=1, stop=len(x) + 1))
    ax1.tick_params(axis='y', labelcolor=color)
    plt.savefig(f"{path_to_look}/{model_name}/all_tasks_accuracy.png")
    plt.show()
    plt.close()
    for j, task in enumerate(tasks):
        y1 = scores_per_task[task]
        y2 = loss[task]
        y2 = movingaverage(y2, epoch_length)
        x2 = [i + 1 for i in range(len(y2))]
        fig = plt.figure()
        ax = fig.add_subplot(111, label="1")
        ax2 = fig.add_subplot(111, label="2", frame_on=False)
        color = 'tab:blue'
        ax.set_xlabel(f'Epoch ({epoch_length} steps in 1 epoch)')
        ax.set_ylabel('Accuracy (major) (%)', color=color)
        x_ax = ax.axes.get_xaxis()
        x_ax.set_major_locator(MaxNLocator(integer=True))
        ax.plot(x, y1, color=color, marker="o")
        ax.set_xticks(np.arange(start=1, stop=len(x) + 1))
        ax.tick_params(axis='y', labelcolor=color)
        color = 'tab:red'
        ax2.plot(x2, y2, color=color)
        ax2.xaxis.tick_top()
        ax2.yaxis.tick_right()
        ax2.set_xlabel(f'Steps ({epoch_length} steps in one epoch)')
        ax2.set_ylabel(f'{task.upper()} training loss (moving average)', color=color)
        ax2.xaxis.set_label_position('top')
        ax2.yaxis.set_label_position('right')
        ax2.tick_params(axis='y', colors=color)
        plt.savefig(f"{path_to_look}/{model_name}/{task}_acc_loss.png")
        plt.show()
        plt.close()

        x2 = [i + 1 for i in range(len(y2))]
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel(f'Steps ({epoch_length} steps in 1 epoch)')
        ax1.set_ylabel(f'{task} loss (moving average)', color=color)
        ax1.plot(x2, y2, color=color)
        plt.savefig(f"{path_to_look}/{model_name}/loss_MA_{task}.png")
        plt.show()
        plt.close()
        y2 = loss[task]
        x2 = [i + 1 for i in range(len(y2))]
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel(f'Steps ({epoch_length} steps in 1 epoch)')
        ax1.set_ylabel(f'{task} loss', color=color)
        ax1.plot(x2, y2, color=color)
        plt.savefig(f"{path_to_look}/{model_name}/loss_{task}.png")
        plt.show()
        plt.close()