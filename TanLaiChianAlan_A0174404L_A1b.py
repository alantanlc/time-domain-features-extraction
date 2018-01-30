import numpy as np
import scipy.io.wavfile as w


def compute_rms(data):
    return np.sqrt(1/len(data) * sum(data**2))


def compute_par(data):
    return max(abs(data)) / compute_rms(data)


def compute_zcr(data):
    data_mul = data[1:len(data)] * data[0:len(data)-1]
    data_mul[data_mul >= 0] = 0
    data_mul[data_mul < 0] = 1
    return np.mean(data_mul)


def compute_mad(data):
    data_med = np.median(data)
    return np.median(abs(data - data_med))


def compute_mean_ad(data):
    return np.mean(abs(data - np.mean(data)))


def compute_uncorrected_std(data):
    return np.sqrt(sum((data - np.mean(data))**2) / (len(data)))


# Open ARFF file to write to
filename = "TanLaiChianAlan_A0174404L_A1b.arff"
arff_file = open(filename, 'w')

# Write header to ARFF file
arff_file.write("@RELATION music_speech\n@ATTRIBUTE RMS_MEAN NUMERIC\n@ATTRIBUTE PAR_MEAN NUMERIC\n@ATTRIBUTE ZCR_MEAN NUMERIC\n@ATTRIBUTE MAD_MEAN NUMERIC\n@ATTRIBUTE MEAN_AD_MEAN NUMERIC\n@ATTRIBUTE RMS_STD NUMERIC\n@ATTRIBUTE PAR_STD NUMERIC\n@ATTRIBUTE ZCR_STD NUMERIC\n@ATTRIBUTE MAD_STD NUMERIC\n@ATTRIBUTE MEAN_AD_STD NUMERIC\n@ATTRIBUTE class {music,speech}\n\n@DATA\n");

# Read ground truth data set
gt_file_name = "music_speech.mf"
gt_file = open(gt_file_name, 'r')
files = gt_file.readlines()
gt_file.close()
for i in range(len(files)):
    files[i] = files[i].strip('\n')
    files[i] = files[i].split('\t')

# Iterate through files and perform computation
for j in files:
    # Read wav file
    [rate, data] = w.read(j[0])

    # Convert to floats by dividing 32768.0
    data = data / 32768.0

    # Split data into 1290 buffers of length 1024 with 50% overlap (or a hop size of 512)
    num_buffers = 1290  # rows
    buffer_size = 1024  # columns
    hop_size = 512
    buffer_data = np.zeros((num_buffers, buffer_size))
    for i in range(num_buffers):
        buffer_data[i] = data[i*hop_size:i*hop_size+buffer_size]

    # Compute features
    num_features = 5
    features = np.zeros((num_buffers, num_features))
    for i in range(num_buffers):
        features[i, 0] = compute_rms(buffer_data[i])
        features[i, 1] = compute_par(buffer_data[i])
        features[i, 2] = compute_zcr(buffer_data[i])
        features[i, 3] = compute_mad(buffer_data[i])
        features[i, 4] = compute_mean_ad(buffer_data[i])

    # Compute mean and uncorrected sample standard deviation for each feature
    result = np.zeros(num_features*2)
    for i in range(num_features):
        result[i] = np.mean(features[:, i])
        result[i+num_features] = compute_uncorrected_std(features[:, i])

    # String output
    output = "{:0.6f},{:0.6f},{:0.6f},{:0.6f},{:0.6f},{:0.6f},{:0.6f},{:0.6f},{:0.6f},{:0.6f},{}\n".format(result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8],result[9],j[1])

    # Write to file
    arff_file.write(output)

# Close file
arff_file.close()
