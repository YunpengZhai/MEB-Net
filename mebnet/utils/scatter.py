import numpy as np

def J_scatter(features, labels):
    label_list = list(set(labels))
    feature_centers =[]
    counts = []
    variance_w = 0
    for i in range(len(label_list)):
        if label_list[i]==-1:
            continue
        i_features = features[labels==label_list[i]]
        i_count = i_features.shape[0]
        counts.append(i_count)
        i_center = np.mean(i_features, axis=0, keepdims=True)
        feature_centers.append(i_center)
        i_variance = np.sum(np.power(i_features - i_center, 2),axis=1)
        i_variance = np.sum(i_variance, axis=0)
        variance_w += i_variance
        # np.dot(i_feature - i_center, i_feature - i_center)
    feature_centers = np.concatenate(feature_centers, axis=0)
    centers_center = np.mean(feature_centers, axis=0, keepdims=True)
    counts = np.array(counts)
    variance_b = np.sum(np.power(feature_centers - centers_center, 2),axis=1)
    variance_b = np.sum(counts*variance_b, axis=0)
    return variance_b/variance_w
