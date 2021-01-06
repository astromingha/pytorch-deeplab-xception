from modeling.deeplab import *
from modeling import custom_transforms as tr
from PIL import Image
from torchvision import transforms
# from modeling.utils import  *
from torchvision.utils import make_grid, save_image
import cv2
import torch
irange = range
import numpy as np
from pycocotools import mask as maskUtils
from skimage import draw


def polygon2mask(image_shape, polygon, color=(255,), image=None):
    vertex_col_coords, vertex_row_coords = polygon.T
    fill_row_coords, fill_col_coords = draw.polygon(
        vertex_row_coords, vertex_col_coords, image_shape)
    if image is None:
        mask = np.zeros(image_shape, dtype=np.uint8)
    else:
        mask = image.copy()
    mask[fill_row_coords, fill_col_coords] = color
    return mask

class Inference():
    def __init__(self):
        ## model load ( at the beginning )
        # model_ckpt = 'config/model_best.pth.tar'
        model_ckpt = 'run/NIA/deeplab-resnet_512/model_best.pth.tar'
        self.model = DeepLab(num_classes=5).cuda()
        ckpt = torch.load(model_ckpt)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

    def __call__(self, image):
        ## image load
        # image_path = 'config/radish_1-1_15.jpg'
        self.original_image = image
        image = Image.fromarray(image).convert('RGB')
        # image = Image.open(image).convert('RGB')
        w,h = image.size
        self.w, self.h = w, h


        ## image input processing for inference
        sample = {'image': image, 'label': image}
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=512),
            tr.RandomGaussianBlur(),
            # tr.Normalize(mean=(0.420, 0.452, 0.343), std=(0.194, 0.187, 0.185)),
            tr.Normalize(mean=(0.352, 0.393, 0.325), std=(0.246, 0.257, 0.230)),
            tr.ToTensor()])
        tensor_in = composed_transforms(sample)['image'].unsqueeze(0)
        image= tensor_in.cuda()

        ## inference
        with torch.no_grad():
            output = self.model(image)

        ## post-processing
        output_nparr_mask = torch.max(output[:3], 1)[1].detach().cpu().numpy()[0].astype(np.uint8)
        # output_npmask_resize = cv2.resize(output_nparr_mask, (w, h), interpolation=cv2.INTER_CUBIC)
        contours, hierachy = cv2.findContours(output_nparr_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours, hierachy = cv2.findContours(output_npmask_resize, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        segmentations,contours_new = [],[]

        for contour in contours:
            area = cv2.contourArea(contour)
            # if area <= 10000: continue
            segmentations.append(contour.flatten().tolist())
            contours_new.append(contour)
            # print("area: ",area)
        self.segmentations = segmentations
        self.contours_new = contours_new
        return segmentations

    def drawContour(self, path):
        img = self.original_image
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)  # 512
        for contour in self.contours_new:
            cv2.drawContours(img, [contour], 0, (255, 255, 255), 2)

        overlay_img = cv2.resize(img, img.shape[:-1])
        # cv2.imwrite(path,overlay_img)
        # cv2.imshow('image', overlay_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def maskImg(self, path):
        img = self.original_image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)  # 512
        seg = self.segmentations

        mask = img.copy()
        for segm in seg:
            pts = np.array([segm]).reshape((-1, 2))
            color = (255, 255, 255)
            mask = polygon2mask(img.shape, pts, color=color, image=mask)

        overlay_img = (0.5 * img + 0.5 * mask).astype(np.uint8)
        # cv2.imwrite('result.jpg', overlay_img)

        cv2.imwrite(path, overlay_img)








