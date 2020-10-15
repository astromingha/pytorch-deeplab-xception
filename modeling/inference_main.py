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

class Inference():
    def __init__(self):
        ## model load ( at the beginning )
        model_ckpt = 'config/model_best.pth.tar'
        self.model = DeepLab(num_classes=2).cuda()
        ckpt = torch.load(model_ckpt)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

    def __call__(self, image):
        ## image load
        # image_path = 'config/radish_1-1_15.jpg'
        image = Image.fromarray(image).convert('RGB')
        # image = Image.open(image).convert('RGB')
        w,h = image.size


        ## image input processing for inference
        sample = {'image': image, 'label': image}
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=512),
            tr.Normalize(mean=(0.420, 0.452, 0.343), std=(0.194, 0.187, 0.185)),
            tr.ToTensor()])
        tensor_in = composed_transforms(sample)['image'].unsqueeze(0)
        image= tensor_in.cuda()

        ## inference
        with torch.no_grad():
            output = self.model(image)

        ## post-processing
        output_nparr_mask = torch.max(output[:3], 1)[1].detach().cpu().numpy()[0].astype(np.uint8)
        output_npmask_resize = cv2.resize(output_nparr_mask, (w, h), interpolation=cv2.INTER_CUBIC)
        contours, hierachy = cv2.findContours(output_npmask_resize, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        segmentations,contours_new = [],[]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 10000: continue
            segmentations.append(contour.flatten().tolist())
            contours_new.append(contour)
            print("area: ",area)
        return segmentations, contours_new