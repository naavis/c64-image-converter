import sys
import numpy as np
import skimage.color
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt

# Approximate sRGB values of the C64 color palette
# Source: http://www.pepto.de/projects/colorvic/
srgb_palette = np.array([
    [0.0, 0.0, 0.0],  # Black
    [254.999999878, 254.999999878, 254.999999878],  # White
    [103.681836072, 55.445357742, 43.038096345],  # Red
    [111.932673473, 163.520631667, 177.928819803],  # Cyan
    [111.399725075, 60.720543693, 133.643433983],  # Purple
    [88.102223525, 140.581101312, 67.050415368],  # Green
    [52.769271594, 40.296416104, 121.446211753],  # Blue
    [183.892638117, 198.676829993, 110.585717385],  # Yellow
    [111.399725075, 79.245328562, 37.169652483],  # Orange
    [66.932804788, 57.383702891, 0.0],  # Brown
    [153.690586380, 102.553762644, 89.111118307],  # Light red
    [67.999561813, 67.999561813, 67.999561813],  # Dark gray
    [107.797780127, 107.797780127, 107.797780127],  # Gray
    [154.244479632, 209.771445903, 131.584994128],  # Light green
    [107.797780127, 94.106015515, 180.927622164],  # Light blue
    [149.480882981, 149.480882981, 149.480882981],  # Light gray
])


def main(args):
    image_filename = args[0]
    image = skimage.io.imread(image_filename)
    if image.shape[2] == 4:
        # Discard alpha channel
        image = image[:, :, :3]
    target_shape = (200, 320, 3)
    if image.shape != target_shape:
        # Resize image if necessary
        image = skimage.transform.resize(image, target_shape, order=3)

    # Convert sRGB image to Lab image
    image_lab = skimage.color.rgb2lab(image)
    image_lab /= image_lab[:, :, 0].max() / 100.0

    # Convert sRGB palette to Lab palette
    palette_lab = np.squeeze(skimage.color.rgb2lab(np.expand_dims(srgb_palette, 0)))
    palette_lab /= palette_lab[:, 0].max() / 100.0

    # For each pixel find the palette color that minimizes delta-E
    differences = np.zeros((16, *target_shape[:2]))
    for color_index, palette_color in enumerate(palette_lab):
        differences[color_index, :, :] = skimage.color.deltaE_cie76(image_lab, palette_color)
    best_indices = differences.argmin(axis=0)

    # Divide image into 8x8 pixel blocks and make sure each block contains only two colors
    split_indices = blockshaped(best_indices, 8, 8)
    reduced_indices = split_indices.copy()
    for i, block in enumerate(split_indices):
        unique_values, value_counts = np.unique(block, return_counts=True)
        if unique_values.shape[0] > 2:
            # More than two colors per block, must reduce!
            # Sort color indices in descending order based on counts
            unique_values_sorted_by_count = unique_values[np.argsort(value_counts)][::-1]

            # Get Lab coordinates for each index to be reduced and
            # compare to most common and second most common color to see which index to use
            top_indices = unique_values_sorted_by_count[:2]
            differences = np.zeros((2, unique_values_sorted_by_count.shape[0] - 2))
            for top_index in range(0, 2):
                differences[top_index, :] = skimage.color.deltaE_cie76(palette_lab[unique_values_sorted_by_count[2:]],
                                                                       palette_lab[top_indices[top_index]])

            replacement_indices = top_indices[np.argmin(differences, axis=0)]

            # Create block with reduced colors
            for index_in_array, color_to_reduce in enumerate(unique_values_sorted_by_count[2:]):
                reduced_indices[i, block == color_to_reduce] = replacement_indices[index_in_array]

    reduced_indices = unblockshaped(reduced_indices, *(target_shape[:2]))
    reduced_image = convert_index_image_to_color_image(reduced_indices, srgb_palette)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=100)
    for i, image_to_show in enumerate([image, reduced_image]):
        ax[i].imshow(image_to_show, interpolation='nearest')
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
    fig.tight_layout()
    plt.show()


def convert_index_image_to_color_image(indices, palette):
    """
    Convert an NxM indexed color image to an NxMx3 RGB color image.
    :param indices: NxM Numpy array of integers, corresponding to the palette indices.
    :param palette: Lx3 Numpy array of floats, corresponding to the palette colors.
    :return: NxMx3 color image as a Numpy array.
    """
    target_shape = (*indices.shape, 3)
    image = np.zeros(target_shape)
    for i in range(0, palette.shape[0]):
        image[indices == i] = srgb_palette[i, :]

    image /= image.max()
    return image


# blockshaped and unblockshaped courtesy of unutbu: http://stackoverflow.com/a/16873755/4350204
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h // nrows, -1, nrows, ncols)
            .swapaxes(1, 2)
            .reshape(h, w))


if __name__ == '__main__':
    main(sys.argv[1:])
