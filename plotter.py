import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy import stats


files = ['env1.csv', 'env2.csv', 'env3.csv']
folders = ['results/', 'results/increase/', 'results/punish/', 'results/punish_comp/']
num_ex = 8
len_ex = 10

fig, axs = plt.subplots(len(files), len(folders)+1)



def convert_from_reward_to_binary(datum):
    new_datum = []
    for d in datum:
        d = int(d)
        new_d = 0.75
        if d == 100:
            new_d = 1
        elif d == -100:
            new_d = 0
        new_datum.append(new_d)
    return new_datum

stuffs = []


for i, filename in enumerate(files):
    stuffs.append([])
    for j, folder in enumerate(folders):
        with open(folder+filename, "r") as f:
            human_data = []
            model_pred = []
            for _ in range(num_ex):
                datum = convert_from_reward_to_binary(f.readline().split(", "))
                human_data.append(datum)
                pred = convert_from_reward_to_binary(f.readline().split(", "))
                model_pred.append(pred)
            clusters = KMeans(n_clusters=2).fit_predict(human_data)

            cluster_human = [[], []]
            cluster_model = [[], []]
            for idx, cluster in enumerate(clusters):
                cluster_human[cluster].append(human_data[idx])
                cluster_model[cluster].append(model_pred[idx])
            
            mean_human_overall = np.mean(human_data, axis=0)
            mean_human_cluster0 = np.mean(cluster_human[0], axis=0)
            mean_human_cluster1 = np.mean(cluster_human[1], axis=0)
            mean_model_overall = np.mean(model_pred, axis=0)
            mean_model_cluster0 = np.mean(cluster_model[0], axis=0)
            mean_model_cluster1 = np.mean(cluster_model[1], axis=0)
            if np.mean(mean_human_cluster0) > np.mean(mean_human_cluster1):
                mean_human_cluster0, mean_human_cluster1 = mean_human_cluster1, mean_human_cluster0
                mean_model_cluster0, mean_model_cluster1 = mean_model_cluster1, mean_model_cluster0
                cluster_human[0], cluster_human[1] = cluster_human[1], cluster_human[0]
                cluster_model[0], cluster_model[1] = cluster_model[1], cluster_model[0]
            
            mean_models = [mean_model_overall, mean_model_cluster0, mean_model_cluster1]
            mean_human = [mean_human_overall, mean_human_cluster0, mean_human_cluster1]
            colors = ['ro-', 'go-', 'bo-']

            if j == 0:
                for d, c in zip(mean_human, colors):
                    axs[i, j].plot(range(len_ex), d, c, markersize=3, linewidth=1) 
                stuffs[i].append(human_data)

            for d, c in zip(mean_models, colors):
                axs[i, j+1].plot(range(len_ex), d, c, markersize=3, linewidth=1) 
            stuffs[i].append(model_pred)


            

for ax in axs.flat:
    ax.label_outer()

axs[0,0].set_title('Human Data')
axs[0,1].set_title('CC')
axs[0,2].set_title('COMP2')
axs[0,3].set_title('CCP')
axs[0,4].set_title('COMP2+CCP')
axs[0,0].set(ylabel='Environment 1')
axs[1,0].set(ylabel='Environment 2')
axs[2,0].set(ylabel='Environment 3')
fig.text(0.02, 0.5, 'Reached Goal Jointly', va='center', rotation='vertical')
fig.text(0.5, 0.04, 'Trial #', ha='center')

plt.savefig("results/avg.png")


#histogram
plt.figure()
fig, axs = plt.subplots(len(files),len(folders)+1)
percent_collabs = []
for i in range(len(stuffs)):
    percent_collabs.append([])
    for j in range(len(stuffs[i])):
        data = stuffs[i][j]
        percent_collab = np.mean(data, axis=1)
        percent_collabs[i].append(percent_collab)
        axs[i,j].hist(percent_collab, bins=10, facecolor='gray', ec='black')

for ax in axs.flat:
    ax.label_outer()

axs[0,0].set_title('Human Data')
axs[0,1].set_title('CC')
axs[0,2].set_title('COMP2')
axs[0,3].set_title('CCP')
axs[0,4].set_title('COMP2+CCP')
axs[0,0].set(ylabel='Environment 1')
axs[1,0].set(ylabel='Environment 2')
axs[2,0].set(ylabel='Environment 3')
fig.text(0.02, 0.5, '# of Examples', va='center', rotation='vertical')
fig.text(0.5, 0.04, '% Cooperation', ha='center')

plt.savefig("results/histo.png")


#scatter
plt.figure()
fig, axs = plt.subplots(len(files),len(folders))
for i in range(len(stuffs)):
    for j in range(1, len(stuffs[i])):
        model_collab = percent_collabs[i][j]
        human_collab = percent_collabs[i][0]
        axs[i,j-1].scatter(model_collab, human_collab)
        _, _, r, _, _ = stats.linregress(model_collab, human_collab)
        print((i,j-1, round(r,2)))

for ax in axs.flat:
    ax.label_outer()

axs[0,0].set_title('CC')
axs[0,1].set_title('COMP2')
axs[0,2].set_title('CCP')
axs[0,3].set_title('COMP2+CCP')
axs[0,0].set(ylabel='Environment 1')
axs[1,0].set(ylabel='Environment 2')
axs[2,0].set(ylabel='Environment 3')
fig.text(0.02, 0.5, 'Data Cooperation %', va='center', rotation='vertical')
fig.text(0.5, 0.04, 'Model Cooperation %', ha='center')

plt.setp(axs, ylim=(0, 1))

plt.savefig("results/scatter.png")

            


