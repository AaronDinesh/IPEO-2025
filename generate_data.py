import numpy as np
import pandas as pd


def process_env(df: pd.DataFrame, mean: np.ndarray = np.empty(0), std: np.ndarray = np.empty(0)):
    # Remove linearly depedent columns
    data = df.drop(columns=["surveyId", "Bio3", "Bio5", "Bio6"]).to_numpy()

    # Normzlize features
    if mean.size == 0:
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)

    return (data - mean) / std, mean, std


def get_timeseries(data):
    """
    Extract timeseries data with shape (T, 4)
    """
    return data.reshape(-1, 40).transpose()


def process_timeseries(
    df: pd.DataFrame, mean: np.ndarray = np.empty(0), std: np.ndarray = np.empty(0)
):
    """
    Create features from timeseries
    """
    # Remove index
    df.drop(columns="surveyId", inplace=True)

    # Reshape to get correct structure
    N, T = df.shape
    landsat_clean = np.zeros((N, T // 4, 4))

    for i, row in df.iterrows():
        landsat_clean[i] = get_timeseries(row.values)

    # Add spectral indices
    landsat_clean = add_spectral_indices(landsat_clean)

    # Normalize features
    if mean.size == 0:
        mean = landsat_clean.mean(axis=(0, 1), keepdims=True)
        std = landsat_clean.std(axis=(0, 1), keepdims=True)

    return (landsat_clean - mean) / (std + 1e-8), mean, std


def process_images(
    images: np.ndarray, mean: np.ndarray = np.empty(0), std: np.ndarray = np.empty(0)
):
    if mean.size == 0:
        mean = images.mean(axis=(0, -2, -1), keepdims=True)
        std = images.std(axis=(0, -2, -1), keepdims=True)

    return (images - mean) / (std + 1e-8), mean, std


"""
Spectral indices helpers
"""


def add_EVI(data):
    """
    Enhanced vegetation index (EVI) = (NIR - R) / (NIR + 6*R - 7.5*B + 1)
    """
    evi = (data[..., 3] - data[..., 0]) / (
        data[..., 3] + 6 * data[..., 0] - 7.5 * data[..., 2] + 1 + 1e-8
    )

    return np.concatenate((data, evi[..., None]), axis=-1)


def add_SAVI(data, L=0.5):
    """
    SAVI = ((NIR - R)(1 + L)) / (NIR + R + L) with L = 0.5
    """
    savi = ((data[..., 3] - data[..., 0]) * (1 + L)) / (data[..., 3] + data[..., 0] + L)

    return np.concatenate((data, savi[..., None]), axis=-1)


def add_GNDVI(data):
    gndvi = (data[..., 3] - data[..., 1]) / (data[..., 3] + data[..., 1] + 1e-8)

    return np.concatenate((data, gndvi[..., None]), axis=-1)


def add_spectral_indices(data, savi=True, evi=True, gnvdi=True):
    if savi:
        data = add_SAVI(data)
    if evi:
        data = add_EVI(data)
    if gnvdi:
        data = add_GNDVI(data)

    return data


# Change path according to data location

# Process env data
env = pd.read_csv("data/env_variables_training.csv")
env_final, env_mean, env_std = process_env(env)

# Process timeseries
landsat = pd.read_csv("data/landsat_timeseries_training.csv")
landsat_final, landsat_mean, landsat_std = process_timeseries(landsat)

# Process images
images_final = np.load("data/satellite_patches_training.npy")  # shape: (N, C, W, H)
print(images_final.shape)
# images_final, images_mean, images_std = process_images(images)

# Load labels
labels = np.load("data/species_data_training.npy")
print(labels.shape)
print(labels[0])

# # Save files
# np.savez(
#     "data/training_set.npz",
#     env=env_final,
#     landsat=landsat_final,
#     images=images_final,
#     labels=labels,
# )

# # Process env data
# env = pd.read_csv("data/env_variables_test.csv")
# env_final_test, _, _ = process_env(env, mean=env_mean, std=env_std)

# # Process timeseries
# landsat = pd.read_csv("data/landsat_timeseries_test.csv")
# landsat_final_test, _, _ = process_timeseries(landsat, mean=landsat_mean, std=landsat_std)

# # Process images
# images_final_test = np.load("data/satellite_patches_test.npy")
# # images_final_test, _, _ = process_images(images, mean=images_mean, std=images_std)

# # Load labels
# labels_test = np.load("data/species_data_test.npy")

# # Save files
# np.savez(
#     "data/validation_set.npz",
#     env=env_final_test,
#     landsat=landsat_final_test,
#     images=images_final_test,
#     labels=labels_test,
# )
