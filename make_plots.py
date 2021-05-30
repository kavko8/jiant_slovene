import matplotlib.pyplot as plt
import os
import jsonlines
from matplotlib.ticker import MaxNLocator
import numpy as np


def create(tasks, model_name, path_to_look, num_epochs):
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
                    score = float("%.2f" % (obj["score"]*100))
                    scores.append(score)
                    for task in tasks:
                        task_score = float("%.2f" % (obj["metrics"][task]["major"]*100))
                        scores_per_task[task].append(task_score)

                #x = [i+1 for i in range(num_epochs)]
                #plt.plot(x, scores, marker="o")

                #plt.show()

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
                    epoch_length = len(values)//num_epochs
                    for j, i in enumerate(values):
                        if ((j+1) % epoch_length) == 0:
                            suma = sum(values[j-epoch_length+1:j+1]) / epoch_length
                            suma = float("%.3f" % suma)
                            epoch_list.append(suma)
                    avg_loss_per_epochs[task] = epoch_list

                    #x = [i+1 for i in range(num_epochs)]
                    #plt.plot(x, epoch_list, marker="o")

                    #plt.show()

    x = [i+1 for i in range(num_epochs)]
    y1 = scores
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (major) (%)', color=color)
    x_ax = ax1.axes.get_xaxis()
    x_ax.set_major_locator(MaxNLocator(integer=True))
    ax1.plot(x, y1, color=color, marker="o")
    ax1.set_xticks(np.arange(start=1, stop=len(x)+1))
    ax1.tick_params(axis='y', labelcolor=color)
    plt.savefig(f"{path_to_look}/{model_name}/all_tasks_accuracy.png")
    plt.show()
    plt.close()
    for j, task in enumerate(tasks):
        y1 = scores_per_task[task]
        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel(f'{task} acc (major) (%)', color=color)
        x_ax = ax1.axes.get_xaxis()
        x_ax.set_major_locator(MaxNLocator(integer=True))
        ax1.plot(x, y1, color=color, marker="o")
        ax1.set_xticks(np.arange(start=1, stop=len(x) + 1))
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        values = avg_loss_per_epochs[task]
        color = 'tab:red'
        ax2.set_ylabel(f"{task} loss", color=color)  # we already handled the x-label with ax1
        x_ax2 = ax2.axes.get_xaxis()
        x_ax2.set_major_locator(MaxNLocator(integer=True))
        ax2.plot(x, values, color=color, marker="o")
        ax2.set_xticks(np.arange(start=1, stop=len(x) + 1))
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(f"{path_to_look}/{model_name}/{task}_acc_loss.png")
        plt.show()
        plt.close()
        x2 = [i + 1 for i in range(len(loss[task]))]
        y2 = loss[task]
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Steps')
        ax1.set_ylabel(f'Loss {task}', color=color)
        ax1.plot(x2, y2, color=color)
        plt.savefig(f"{path_to_look}/{model_name}/loss_{task}.png")
        plt.show()
        plt.close()
