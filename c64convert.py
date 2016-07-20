import sys
import numpy as np
import skimage.color
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt

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
        image = skimage.transform.rescale(image, target_shape)

    # Convert sRGB image to Lab image
    image_lab = skimage.color.rgb2lab(image)
    image_lab /= image_lab[:, :, 0].max() / 100.0

    # Convert sRGB palette to Lab palette
    palette_lab = np.squeeze(skimage.color.rgb2lab(np.expand_dims(srgb_palette, 0)))
    palette_lab /= palette_lab[:, 0].max() / 100.0

    # For each pixel find the palette color that minimizes delta-E
    differences = np.zeros((16, 200, 320))
    for color_index, palette_color in enumerate(palette_lab):
        differences[color_index, :, :] = skimage.color.deltaE_cie76(image_lab, palette_color)
    best_indices = differences.argmin(axis=0)

    best_match_image = np.zeros(target_shape)
    for i in range(0, srgb_palette.shape[0]):
        best_match_image[best_indices == i] = srgb_palette[i, :]

    best_match_image /= best_match_image.max()
    plt.imshow(best_match_image, interpolation="nearest")
    plt.show()

    # TODO: Divide image into 8x8 pixel blocks and make sure each block contains only two colors


def find_best_palette_index(lab_pixel, lab_palette):
    differences = np.array([skimage.color.deltaE_cie76(lab_pixel, color) for color in lab_palette])
    return differences.argmin()


if __name__ == '__main__':
    main(sys.argv[1:])
