import numpy as np
import scipy.io.wavfile as w
import scipy.signal as sig
import scipy.fftpack as f


def compute_sc(data_abs):
    return sum(np.arange(len(data_abs)) * data_abs) / sum(data_abs)


def compute_sro(data_abs, L):
    # Compute L energy
    rhs = L * sum(data_abs)

    # Find smallest bin index R such that L energy is less than the sum of it
    r = -1
    for i in range(len(data_abs)):
        x = data_abs[0:i+1]
        if sum(x) >= rhs:
            r = i
            break

    return r


def compute_sfm(data_abs):
    geo_mean = np.exp(np.mean(np.log(data_abs)))
    arith_mean = np.mean(data_abs)
    return geo_mean / arith_mean


def compute_rms(data_abs):
    return np.sqrt(1/len(data_abs) * sum(data_abs**2))


def compute_par(data_abs):
    return max(abs(data_abs)) / compute_rms(data_abs)


def compute_sf(data_abs, data_abs_prev):
    h = data_abs - data_abs_prev
    h[h < 0] = 0
    return sum(h)


def compute_uncorrected_std(data):
    return np.sqrt(sum((data - np.mean(data))**2) / (len(data)))


# Open ARFF file to write to
filename = "TanLaiChianAlan_A0174404L_A2.arff"
arff_file = open(filename, 'w')

# Write header to ARFF file
arff_file.write("@RELATION music_speech\n@ATTRIBUTE SC_MEAN NUMERIC\n@ATTRIBUTE SRO_MEAN NUMERIC\n@ATTRIBUTE SFM_MEAN NUMERIC\n@ATTRIBUTE PARFFT_MEAN NUMERIC\n@ATTRIBUTE FLUX_MEAN NUMERIC\n@ATTRIBUTE SC_STD NUMERIC\n@ATTRIBUTE SRO_STD NUMERIC\n@ATTRIBUTE SFM_STD NUMERIC\n@ATTRIBUTE PARFFT_STD NUMERIC\n@ATTRIBUTE FLUX_STD NUMERIC\n@ATTRIBUTE class {music,speech}\n\n@DATA\n");

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
    hamming = sig.hamming(buffer_size)
    data_abs_prev = np.zeros((int)(buffer_size/2)+1)
    for i in range(num_buffers):
        # Apply hamming Window
        data_hamming = buffer_data[i] * hamming

        # Discrete Fourier Transform (Discard the negative frequencies)
        data_fft = f.fft(data_hamming)
        data_fft = data_fft[0:(int)(buffer_size/2)+1]

        # Convert spectrum to absolute values
        data_abs = np.abs(data_fft)

        # Compute Spectral Centroid (SC)
        features[i, 0] = compute_sc(data_abs)

        # Compute Spectral Roll-Off (SRO)
        features[i, 1] = compute_sro(data_abs, 0.85)

        # Compute Spectral Flatness Measure (SFM)
        features[i, 2] = compute_sfm(data_abs)

        # Compute Peak-to-average ratio (PARFFT)
        features[i, 3] = compute_par(data_abs)

        # Compute Spectral Flux (SF)
        features[i, 4] = compute_sf(data_abs, data_abs_prev)

        # Update data_abs_prev
        data_abs_prev = data_abs

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
