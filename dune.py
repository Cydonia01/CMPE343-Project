import pandas as pd
import numpy  as np

# load the data
data = pd.read_csv("detection_data.csv")

dist = data['Distance'].values
amp = data['Amplitude'].values
detections = data['Detection'].values

# convert detect to 1, no detect to 0
bin_detections = np.array([1 if d == 'Detect' else 0 for d in detections])

# split into two groups
a_detect = amp[bin_detections == 1]
a_no_detect = amp[bin_detections == 0]

d_detect = dist[bin_detections == 1]
d_no_detect = dist[bin_detections == 0]

# calculate the mean and standard deviation of the amplitude and distance for each group
a_detect_mean = np.mean(a_detect)
a_detect_std = np.std(a_detect)
d_detect_mean = np.mean(d_detect)
d_detect_std = np.std(d_detect)

a_no_detect_mean = np.mean(a_no_detect)
a_no_detect_std = np.std(a_no_detect)
d_no_detect_mean = np.mean(d_no_detect)
d_no_detect_std = np.std(d_no_detect)

# compute the probabilities
p_detect = np.mean(bin_detections)
p_no_detect = 1 - p_detect

def normal_pdf(x, mean, std):
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

def classification(a_value, d_value):
    p_detect_a = normal_pdf(a_value, a_detect_mean, a_detect_std)
    p_detect_d = normal_pdf(d_value, d_detect_mean, d_detect_std)
    joint_detect = p_detect_a * p_detect_d * p_detect

    p_no_detect_a = normal_pdf(a_value, a_no_detect_mean, a_no_detect_std)
    p_no_detect_d = normal_pdf(d_value, d_no_detect_mean, d_no_detect_std)
    joint_no_detect = p_no_detect_a * p_no_detect_d * p_no_detect

    return joint_detect > joint_no_detect

# classificate for all data points
predictions = np.array([classification(a, d) for a, d in zip(amp, dist)])

accuracy = np.mean(predictions == bin_detections)
print(f"Accuracy: {accuracy:.2f}")

# Processing of new data
new_data = pd.read_csv('detection_data_extra.csv')

new_dist = new_data['Distance'].values
new_amp = new_data['Amplitude'].values
new_detections = new_data['Detection'].values

bin_new_detections = np.array([1 if d == 'Detect' else 0 for d in new_detections])

predictions_new = np.array([classification(a, d) for a, d in zip(new_amp, new_dist)])

accuracy_new = np.mean(predictions_new == bin_new_detections)
print(f"Accuracy new data: {accuracy_new:.2f}")