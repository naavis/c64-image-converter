import sys
import numpy as np
import skimage.color
import skimage.io
import skimage.transform

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

    # TODO: For each pixel find the palette color that minimizes delta-E
    # TODO: Process each 8x8 block to use only 2 colors per block


if __name__ == '__main__':
    main(sys.argv[1:])
