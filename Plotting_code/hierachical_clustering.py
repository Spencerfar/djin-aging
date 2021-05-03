import numpy as np
import argparse
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import laplace

from scipy.cluster.hierarchy import dendrogram as set_link_color_palette, dendrogram
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
cm = plt.get_cmap('tab10')


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    
    dendrogram(linkage_matrix, **kwargs)


parser = argparse.ArgumentParser('Hierarchical clustering')
parser.add_argument('--job_id', type=int)
parser.add_argument('--epoch', type=int)
args = parser.parse_args()

palette = [mpl.colors.to_hex(cm(k)[:3]) for k in range(10)]
del palette[5]
del palette[-3]

network = np.load('../Analysis_Data/network_weights_job_id%d_epoch%d.npy'%(args.job_id, args.epoch))


A = np.nan_to_num(network, nan=0.0)
A = np.abs(A)
A=(A.T+A)/2.0
np.fill_diagonal(A, np.max(A)*1.05)     
A = (np.max(A) - A)

clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='precomputed', linkage='average')
clusering = clustering.fit(A)


deficits = ['Gait speed', 'Dom Grip strength', 'Non-dom grip str', 'ADL score','IADL score', 'Chair rises','Leg raise','Full tandem stance', 'Self-rated health', 'Eyesight','Hearing', 'General functioning', 'Diastolic blood pressure', 'Systolic blood pressure', 'Pulse', 'Triglycerides','C-reactive protein','HDL cholesterol','LDL cholesterol','Glucose','IGF-1','Hemoglobin','Fibrinogen','Ferritin', 'Total cholesterol', r'White blood cell count', 'MCH', 'Glycated hemoglobin', 'Vitamin-D']

mpl.rcParams['lines.linewidth'] = 3.5


fig,ax = plt.subplots(figsize=(3,10))

plot_dendrogram(clustering, labels = deficits, orientation='right', color_threshold = 0.112, above_threshold_color="darkgrey",)

ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(bottom=False, labelsize=12)
ax.set_xticklabels([])
#ax.set_yticklabels([])

plt.tight_layout()
plt.subplots_adjust(bottom=0.35)
plt.savefig('../Plots/hierarchical_network_clustering_job_id%d_epoch%d.pdf'%(args.job_id, args.epoch))
