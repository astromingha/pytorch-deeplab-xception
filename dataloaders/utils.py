import matplotlib.pyplot as plt
import numpy as np
import torch

def decode_seg_map_sequence(label_masks, dataset='cityscapes'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal' or dataset == 'coco':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 43#37
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        # r[label_mask == ll] = label_colours[ll, 0]
        # g[label_mask == ll] = label_colours[ll, 1]
        # b[label_mask == ll] = label_colours[ll, 2]
        r[label_mask == ll] = label_colours[ll, 2]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 0]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def get_cityscapes_labels():
    return np.array([
        [194, 230, 254], [111, 193, 223], [132, 132, 192], [184, 131, 237], [164, 176, 223], [138, 113, 246], [254, 38, 229],
        [81, 50, 197], [78, 4, 252], [42, 65, 247], [0, 0, 115], [18, 177, 246], [0, 122, 255], [27, 88, 199], [191, 255, 255],
        [168, 230, 244], [102, 249, 247], [10, 228, 245], [115, 220, 223], [44, 177, 184], [18, 145, 184], [0, 100, 170],
        [44, 160, 51], [64, 79, 10], [51, 102, 51], [148, 213, 161], [90, 228, 128], [90, 176, 113], [51, 126, 96], [208, 167, 180],
        [153, 116, 153], [162, 30, 124], [236, 219, 193], [202, 197, 171], [165, 182, 171],[138, 90, 88], [172, 181, 123],
        [255, 242, 159], [255, 167, 62], [255, 109, 93], [255, 57, 23], [0, 0, 0], [255, 255, 255]
        ])

    # return np.array([
    #     [128, 64, 128],
    #     [244, 35, 232],
    #     [70, 70, 70],
    #     [102, 102, 156],
    #     [190, 153, 153],
    #     [153, 153, 153],
    #     [250, 170, 30],
    #     [220, 220, 0],
    #     [107, 142, 35],
    #     [152, 251, 152],
    #     [0, 130, 180],
    #     [220, 20, 60],
    #     [255, 0, 0],
    #     [0, 0, 142],
    #     [0, 0, 70],
    #     [0, 60, 100],
    #     [0, 80, 100],
    #     [0, 0, 230],
    #     [119, 11, 32]])


    # return np.array([
    #     [252, 0, 189],
    #     [244, 179, 252],
    #     [189, 226, 252],
    #     [252, 189, 184],
    #     [179, 252, 196],
    #     [145, 38, 153],
    #     [47, 153, 43],
    #     [181, 85, 40],
    #     [64, 148, 184],
    #     [171, 164, 43],
    #     [138, 186, 139],
    #     [157, 156, 214],
    #     [186, 219, 224],
    #     [211, 212, 152],
    #     [132, 156, 181],
    #     [176, 176, 176],
    #     [255, 187, 0],
    #     [153, 255, 0],
    #     [31, 57, 255],
    #     [255, 0, 0],
    #     [115, 77, 42],
    #     [201, 137, 52],
    #     [0, 0, 0],
    #     [91, 63, 176],
    #     [0, 97, 0],
    #     [122, 171, 0],
    #     [255, 255, 0],
    #     [255, 153, 0],
    #     [255, 34, 0],
    #     [146, 181, 96],
    #     [136, 86, 245],
    #     [176, 74, 89],
    #     [117, 180, 235],
    #     [250, 152, 242],
    #     [45, 61, 128],
    #     [83, 230, 203],
    #     [107, 245, 86]])



def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])