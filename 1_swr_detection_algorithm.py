#SWR DETECTION ALGORITHM


import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
from skimage.measure import label, regionprops



def detect_SWR(eeg, eeg_fs, ied_bool, data_dropout,rippleBand=[80, 150], gaussFilt=10.0, zThresh=3, durTresh=(0.03, 0.15) ):
    
    #remove IED times and data dropout
    buffer_samples = int(0.5 * eeg_fs)
    valid_regions_1 = np.ones(eeg.shape[1], dtype=bool)


    ied_indices = np.where(ied_bool[0])[0]  # Extract IED indices
    for idx in ied_indices:
        start = max(0, idx - buffer_samples)
        end = min(valid_regions_1.shape[0], idx + buffer_samples + 1)
        valid_regions_1[start:end] = False


    for start, end in data_dropout:
        valid_regions_1[start:end] = False

    #filter whole EEG signal in ripple band
    b = signal.firwin(401, np.array(rippleBand)/ (0.5 * eeg_fs), pass_zero = False )
    filt_eeg = signal.filtfilt(b, 1, eeg)
    ripple_amp = np.abs(signal.hilbert(filt_eeg))
    
    #gaussian smoothing 
    gaussian_sigma = gaussFilt/ 1000 * eeg_fs
    smoothed_ripple_amp = ndimage.gaussian_filter1d(ripple_amp, sigma = gaussian_sigma)
    
    #z-score it using only the previously computed valid regions
    smoothed_ripple_amp_Z = (smoothed_ripple_amp - np.mean(smoothed_ripple_amp[:, valid_regions_1])) / np.std(smoothed_ripple_amp[:, valid_regions_1])
    
    #get regions above mean
    valid_regions = (smoothed_ripple_amp_Z >= 0) & valid_regions_1

    #dialate and erode to merge neighbouring regions
    min_duration_samples = round(durTresh[0]*eeg_fs)
    structure_element = np.ones((1,min_duration_samples))
    valid_regions = ndimage.binary_dilation(valid_regions, structure = structure_element)
    valid_regions = ndimage.binary_erosion(valid_regions, structure = structure_element)

    #label connected valid regions
    labeled_regions = label(valid_regions)

    #compute region properties
    regions = regionprops(labeled_regions, intensity_image=smoothed_ripple_amp_Z)

    #get duration (area) and peak Z score (max intesity)
    durations = np.array([region.area / eeg_fs for region in regions])
    peak_Z = np.array([region.max_intensity for region in regions])

    #apply duration treshold
    duration_min, duration_max = durTresh

    selected_regions = [
        region for region, dur, peak in zip(regions, durations, peak_Z)
        if duration_min <= dur <= duration_max and peak >= zThresh
    ]

    output = {
        "nEvents": len(selected_regions),
        "durations": np.array([region.area / eeg_fs for region in selected_regions]),
        "rippleTime": [],
        "rippleEEG": [],
        "rippleAmp": []
    }

    for region in selected_regions:
        start, end = region.bbox[1], region.bbox[3] 
        output["rippleTime"].append((region.coords[:, 1] / eeg_fs).tolist())
        output["rippleEEG"].append(ripple_amp[:, start:end])
        output["rippleAmp"].append(smoothed_ripple_amp_Z[:, start:end])
    return output

    
    