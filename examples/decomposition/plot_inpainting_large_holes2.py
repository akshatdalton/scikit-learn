from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn import linear_model


try:  # SciPy >= 0.16 have face in misc
    from scipy.misc import face

    face = face(gray=True)
except ImportError:
    face = sp.face(gray=True)

MISSING_PIXEL = -100

# Convert from uint8 representation with values between 0 and 255 to
# a floating point representation with values between 0 and 1.
face = face.astype("int16")
face = face / 255.0

# im_ascent = sp.misc.ascent().astype(float)
im_ascent = face
height, width = im_ascent.shape

# Distort the image
print("Distorting image...")

distorted = face.copy()
distorted[height // 4 - 50:height // 4 + 50, width // 4 - 50:width // 4 + 50] = MISSING_PIXEL
result = distorted.copy()

# plt.imshow(distorted, vmin=0, vmax=1, cmap=plt.cm.gray, interpolation="nearest")
# plt.show()
# exit(0)


# Extract all reference patches from the left half of the image
print("Extracting reference patches...")
t0 = time()
patch_len = 41
patch_size = (patch_len, patch_len)
data = extract_patches_2d(distorted, patch_size)
data = data.reshape(data.shape[0], -1)
data_len = len(data)

max_atoms = 100
curr_atoms = 0
D = []
for d in data:
    if MISSING_PIXEL not in d:
        curr_atoms
        D.append(d)
        if curr_atoms == max_atoms:
            break

dico = MiniBatchDictionaryLearning(
    n_components=len(D), alpha=1e-4, n_iter=1000, dict_method="lasso_cd"
)

# # Removing bias
# D -= np.mean(D, axis=0)
# D /= np.std(D, axis=0)
# print("done in %.2fs." % (time() - t0))


plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(D[:100]):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r, interpolation="nearest")
    plt.xticks(())
    plt.yticks(())
plt.suptitle(
    "Dictionary learned from im_ascent patches\n"
    + "Train time on %d patches" % (len(data)),
    fontsize=16,
)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

# #############################################################################
# Display the distorted image


def show_with_diff(image, reference, title):
    """Helper function to display denoising"""
    image[image == MISSING_PIXEL] = 0
    plt.figure(figsize=(5, 3.3))
    plt.subplot(1, 2, 1)
    plt.title(title)
    plt.imshow(image, vmin=0, vmax=1, cmap=plt.cm.gray, interpolation="nearest")
    plt.xticks(())
    plt.yticks(())
    plt.subplot(1, 2, 2)
    difference = image - reference

    plt.title("Difference (norm: %.2f)" % np.sqrt(np.sum(difference ** 2)))
    plt.imshow(
        difference, vmin=-0.5, vmax=0.5, cmap=plt.cm.PuOr, interpolation="nearest"
    )
    plt.xticks(())
    plt.yticks(())
    plt.suptitle(title, size=16)
    plt.subplots_adjust(0.02, 0.02, 0.98, 0.79, 0.02, 0.2)


show_with_diff(distorted, im_ascent, "Distorted image")

t0 = time()

def get_patch(x, y):
    x_start, x_end = x - patch_len // 2, x + patch_len // 2 + 1
    y_start, y_end = y - patch_len // 2, y + patch_len // 2 + 1
    return result[x_start:x_end, y_start:y_end].copy()


def fill_patch(x, y, patch_vector):
    x_start, x_end = x - patch_len // 2, x + patch_len // 2 + 1
    y_start, y_end = y - patch_len // 2, y + patch_len // 2 + 1
    result[x_start:x_end, y_start:y_end] = patch_vector.reshape(*patch_size)


def naive_high_priority_pixel():
    curr_priority = -1
    curr_index = (-1, -1)
    curr_patch = None
    boundary_len = 0

    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            if result[x, y] == MISSING_PIXEL:
                patch = get_patch(x, y).ravel()
                if not (MISSING_PIXEL == patch).all():
                    boundary_len += 1
                    count_px_complete = np.argwhere(patch != MISSING_PIXEL).shape[0]
                    if count_px_complete > curr_priority:
                        curr_priority = count_px_complete
                        curr_index = (x, y)
                        curr_patch = patch
    return curr_index, curr_patch, boundary_len

dico.set_params(transform_algorithm="lasso_cd", transform_alpha=1.0)

D = np.array(D)
curr_index, curr_patch, boundary_len = naive_high_priority_pixel()

while curr_patch is not None:
    print("boundary_len = ", boundary_len)
    corrupted_index = np.where(curr_patch == MISSING_PIXEL)[0]
    non_corrupted_index = np.where(curr_patch != MISSING_PIXEL)[0]
    # print("D.shape = ", D.shape)
    # print("patch.shape = ", curr_patch.shape)
    dico.components_ = D.T[non_corrupted_index].T
    # print("Dcurr.shape = ", dico.components_.shape)
    # print("patchcurr.shape = ", curr_patch[non_corrupted_index].reshape(1, -1).shape)
    code = dico.transform(curr_patch[non_corrupted_index].reshape(1, -1))
    curr_patch[corrupted_index] = np.dot(code, D)[corrupted_index]
    fill_patch(curr_index[0], curr_index[1], curr_patch)

    curr_index, curr_patch, boundary_len = naive_high_priority_pixel()


# patches = data.reshape(data_len, *patch_size)
# reconstructed_image = reconstruct_from_patches_2d(patches, (height, width))
dt = time() - t0
show_with_diff(result, im_ascent, "Recontructed Image" + " (time: %.1fs)" % dt)
plt.show()

exit(0)

# # #############################################################################
# # Extract noisy patches and reconstruct them using the dictionary

# print("Extracting noisy patches... ")
# t0 = time()
# data = extract_patches_2d(distorted, patch_size)
# data = data.reshape(data.shape[0], -1)
# intercept = np.mean(data, axis=0)
# data -= intercept
# print("done in %.2fs." % (time() - t0))

# transform_algorithms = [
#     ("Orthogonal Matching Pursuit\n1 atom", "omp", {"transform_n_nonzero_coefs": 1}),
#     ("Thresholding\n alpha=0.1", "threshold", {"transform_alpha": 0.1}),
#     ("Orthogonal Matching Pursuit\n2 atoms", "omp", {"transform_n_nonzero_coefs": 2}),
#     # ("Orthogonal Matching Pursuit\n8 atoms", "omp", {"transform_n_nonzero_coefs": 8}),
#     # ("Orthogonal Matching Pursuit\n16 atoms", "omp", {"transform_n_nonzero_coefs": 16}),
#     ("Least-angle regression\n4 atoms", "omp", {"transform_n_nonzero_coefs": 4}),
# ]

# reconstructions = {}
# for title, transform_algorithm, kwargs in transform_algorithms:
#     print(title + "...")
#     t0 = time()
#     dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
#     code = dico.transform(data)
#     patches = np.dot(code, D)

#     patches += intercept
#     patches = patches.reshape(len(data), *patch_size)
#     if transform_algorithm == "threshold":
#         patches -= patches.min()
#         patches /= patches.max()
#     reconstructions[title] = reconstruct_from_patches_2d(patches, (height, width))
#     dt = time() - t0
#     print("done in %.2fs." % dt)
#     show_with_diff(reconstructions[title], im_ascent, title + " (time: %.1fs)" % dt)
#     plt.show()

# plt.show()
# # plt.savefig("plot_figure", format='png')
