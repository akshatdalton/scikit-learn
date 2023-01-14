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
distorted[height // 4 - 25:height // 4 + 25, width // 4 - 25:width // 4 + 25] = MISSING_PIXEL
result = distorted.copy()

# plt.imshow(distorted, vmin=0, vmax=1, cmap=plt.cm.gray, interpolation="nearest")
# plt.show()
# exit(0)

def get_patch(x_centre, y_centre, h):
    x_start, x_end = x_centre - h // 2, x_centre + h // 2
    y_start, y_end = y_centre - h // 2, y_centre + h // 2
    return result[x_start:x_end, y_start:y_end].copy()


def fill_patch(x_centre, y_centre, h, values):
    x_start, y_start = x_centre - h // 2, y_centre - h // 2
    x_end, y_end = x_centre + h // 2, y_centre + h // 2
    result[x_start:x_end, y_start:y_end] = values.reshape(h, h)


def get_boundary_pixels(h):
    boundary_pixels = []
    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            if result[x, y] == MISSING_PIXEL:
                curr_patch = get_patch(x, y, h)
                if not (curr_patch.ravel() == MISSING_PIXEL).all():
                    boundary_pixels.append((x, y))

    return boundary_pixels


def naive_high_priority_pixel(boundary_pixels, h):
    curr_priority = -1
    curr_pixel_index = (-1, -1)
    curr_patch = None
    for pixel in boundary_pixels:
        patch = get_patch(pixel[0], pixel[1], h)
        count_complete_pixel = np.argwhere(patch.ravel() != MISSING_PIXEL).shape[0]
        if count_complete_pixel > curr_priority:
            curr_priority = count_complete_pixel
            curr_pixel_index = pixel
            curr_patch = patch

    return curr_pixel_index, curr_patch


def get_dictionary(h, stride):
    dic = []
    X, Y = distorted.shape
    for i in range(0, X - h, stride):
        for j in range(0, Y - h, stride):
            patch = get_patch(i + h // 2, j + h // 2, h).ravel()
            if not MISSING_PIXEL in patch:
                dic.append(patch)
    return np.array(dic).T


def inpainting(h, stride, alpha, max_iter=60000):
    t_init = time()
    
    D = get_dictionary(h, stride)
    print("D.shape = ", D.shape)
    boundary_pixels = get_boundary_pixels(h)
    print("len bd = ", len(boundary_pixels))

    model = linear_model.Lasso(alpha=alpha, max_iter=max_iter)
    i = 0

    while len(boundary_pixels) > 0:
        print("len bd = ", len(boundary_pixels))
        centre_pixel, Y = naive_high_priority_pixel(boundary_pixels, h)
        Y = Y.ravel()
        train_iter = []
        test_iter = []
        
        for k in range(len(Y)):
            if Y[k] == MISSING_PIXEL:
                test_iter.append(k)
            else:
                train_iter.append(k)

        model.fit(D[train_iter], Y[train_iter])

        Y[test_iter] = model.predict(D[test_iter])
        fill_patch(centre_pixel[0], centre_pixel[1], h, Y)
        boundary_pixels = get_boundary_pixels(h)
        i += 1

    print("elapsed time: {0:.2f} seconds".format(time() - t_init))


inpainting(h=40, stride=7, alpha=1e-4, max_iter=7000)


# plt.imshow(result, vmin=0, vmax=1, cmap=plt.cm.gray, interpolation="nearest")
# plt.show()
# exit(0)

# # Extract all reference patches from the left half of the image
# print("Extracting reference patches...")
# t0 = time()
# patch_size = (7, 7)
# data = extract_patches_2d(distorted, patch_size)
# data = data.reshape(data.shape[0], -1)
# data_len = len(data)

# pure_atoms = []
# for d in data:
#     if MISSING_PIXEL not in d:
#         pure_atoms.append(d)

# # Removing bias
# pure_atoms -= np.mean(pure_atoms, axis=0)
# pure_atoms /= np.std(pure_atoms, axis=0)
# print("done in %.2fs." % (time() - t0))

# # #############################################################################
# # Learn the dictionary from reference patches

# print("Learning the dictionary...")
# t0 = time()
# dico = MiniBatchDictionaryLearning(
#     n_components=len(pure_atoms), alpha=1e-4, n_iter=1000, dict_method="lasso_lars"
# )
# V = dico.fit(pure_atoms).components_
# dt = time() - t0
# print("done in %.2fs." % dt)

# plt.figure(figsize=(4.2, 4))
# for i, comp in enumerate(V[:100]):
#     plt.subplot(10, 10, i + 1)
#     plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r, interpolation="nearest")
#     plt.xticks(())
#     plt.yticks(())
# plt.suptitle(
#     "Dictionary learned from im_ascent patches\n"
#     + "Train time %.1fs on %d patches" % (dt, len(data)),
#     fontsize=16,
# )
# plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

# # #############################################################################
# # Display the distorted image


def show_with_diff(image, reference, title):
    """Helper function to display denoising"""
    plt.figure(figsize=(5, 3.3))
    plt.subplot(1, 2, 1)
    plt.title("Image")
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
show_with_diff(result, im_ascent, "Restored image")

# dico.set_params(transform_algorithm="omp", **{"transform_n_nonzero_coefs": 1})


# def fill_patch(index, patch):
#     patch[patch == MISSING_PIXEL] = 0
#     code = dico.transform(patch.reshape(1, -1))
#     reconstructed_patch = np.dot(code, V)
#     patch_len = patch_size[0]
#     x_start, x_end = index[0] - patch_len // 2, index[0] + patch_len // 2 + 1
#     y_start, y_end = index[1] - patch_len // 2, index[1] + patch_len // 2 + 1
#     result[x_start:x_end, y_start:y_end] = reconstructed_patch.reshape(*patch_size)

# t0 = time()

# boundary_pixel = get_boundary_pixels()
# print("bd len = ", len(boundary_pixel))

# while len(boundary_pixel) > 0:
#     next_patch_index, next_patch = naive_high_priority_pixel(boundary_pixel)
#     fill_patch(next_patch_index, next_patch)
#     boundary_pixel = get_boundary_pixels()

#     print("bd len = ", len(boundary_pixel))


# # patches = data.reshape(data_len, *patch_size)
# # reconstructed_image = reconstruct_from_patches_2d(patches, (height, width))
# dt = time() - t0
# show_with_diff(result, im_ascent, "Recontructed Image" + " (time: %.1fs)" % dt)
# plt.show()

# exit(0)

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
#     patches = np.dot(code, V)

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

plt.show()
# # plt.savefig("plot_figure", format='png')
