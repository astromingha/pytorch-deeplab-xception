from modeling.deeplab import *
from modeling import custom_transforms as tr
from PIL import Image
from torchvision import transforms
# from modeling.utils import  *
from torchvision.utils import make_grid, save_image
import cv2,tqdm,os
import torch
import numpy as np
from pycocotools import mask as maskUtils
from skimage import draw
from dataloaders.utils import decode_seg_map_sequence_
from torch.utils import data
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

class GetData(data.Dataset):
    def __init__(self, crop_size, img_dir):
        self.NUM_CLASSES = 5
        # self.root = img_dir
        self.split = 'test'
        self.crop_size = crop_size
        # self.args = args
        self.files = {}
        self.file_idx = 0

        self.images_base = img_dir
        # self.annotations_base = os.path.join(self.root, 'gtFine', self.split)

        self.files[self.split] = self.recursive_glob(rootdir=self.images_base)#, suffix='.png')

        self.void_classes = [-1]#
        self.valid_classes = [0,1,2,3,4]

        self.class_names = ['0','1','2','3','4']

        self.mean = (0.352, 0.393, 0.325)#(0.242, 0.324, 0.241)
        self.std = (0.246, 0.257, 0.230)#(0.188, 0.190, 0.179)
        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        # if not self.files[split]:
        #     raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))
        #
        # print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        _img = Image.open(img_path).convert('RGB')
        sample = {'image': _img, 'label': _img}
        # sample = {'image': _img, 'label': _img, 'file_idx': np.array([self.file_idx])}
        temp = self.transform_ts(sample)
        temp["file_dir"] = torch.as_tensor(self.file_idx)
        self.file_idx += 1
        return temp#self.transform_ts(sample)

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_ts(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.crop_size),
            tr.Normalize(mean=self.mean, std=self.std),
            tr.ToTensor()])

        return composed_transforms(sample)


class Inference():
    def __init__(self):
        ## model load ( at the beginning )
        model_ckpt = 'run/NIA/deeplab-resnet_512/model_best.pth.tar'
        self.batch_size = 200#16
        self.crop_size = 512
        self.kwargs = {'num_workers': 2, 'pin_memory': True}

        self.model = DeepLab(num_classes=5).cuda()
        ckpt = torch.load(model_ckpt)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

    def __call__(self, image):
        ## image load
        # self.original_image = image
        # image = Image.fromarray(image).convert('RGB')
        # w,h = image.size
        # self.w, self.h = w, h

        ## image input processing for inference
        self.data_tensors = GetData(self.crop_size, image)
        val_loader = data.DataLoader(self.data_tensors, batch_size=self.batch_size, shuffle=False, **self.kwargs)

        segmentations_batch, contours_new_batch = [], []

        for smpl in tqdm.tqdm(val_loader,desc='\r'):
            image = smpl['image']
            if torch.cuda.is_available():
                image = image.cuda()
            else: raise RuntimeError
            with torch.no_grad():
                output_batch = self.model(image)


            for single_img in output_batch.data:
                output_nparr = torch.max(single_img.unsqueeze(0)[:3], 1)[1].detach().cpu().numpy()
                output_nparr_mask_splited = decode_seg_map_sequence_(output_nparr,
                                                                     num_classes=val_loader.dataset.NUM_CLASSES)

                segmentations, contours_new = [], []
                for cls in range(output_nparr_mask_splited[0].shape[-1]): # choose one class from images
                    output_nparr_mask = output_nparr_mask_splited[0][:, :, cls].astype(np.uint8)
                    ## no output resize!! original image must be same as output tensor size!!!!!!!!
                    # output_nparr_mask = cv2.resize(output_nparr_mask, (2048, 2048), interpolation=cv2.INTER_CUBIC)
                    #  if output_nparr_mask.shape[0] != h or output_nparr_mask.shape[1] != w:
                    #     output_nparr_mask = cv2.resize(output_nparr_mask, (w, h), interpolation=cv2.INTER_CUBIC)
                    contours, hierachy = cv2.findContours(output_nparr_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours:
                        area = cv2.contourArea(contour)
                        # if area <= 10000: continue
                        segmentations.append(contour.flatten().tolist()+[cls])
                        contours_new.append(contour)
                        # print("area: ",area)

                segmentations_batch.append(segmentations) #[ image1[ [seg1[contour1[],contour2[],... ], [seg2], ... ], [image2]...]
                contours_new_batch.append(contours_new)
                self.segmentations = segmentations_batch
                self.contours_new = contours_new_batch

        return segmentations_batch

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
        image_files = self.data_tensors.files['test']
        color = [(255, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        seg = self.segmentations

        for idx,img_dir in enumerate(image_files):
            img = cv2.imread(img_dir)
            mask = img.copy()
            for segm in seg[idx]:
                pts = np.array([segm[:-1]]).reshape((-1, 2))
                class_num = segm[-1]

                # if class_num != 0: continue
                mask = polygon2mask(img.shape, pts, color=color[class_num], image=mask)

            overlay_img = (0.5 * img + 0.5 * mask).astype(np.uint8)
            cv2.imwrite(os.path.join(path,os.path.split(img_dir)[-1]), overlay_img)




        #
        # img = self.original_image
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # # img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)  # 512
        # seg = self.segmentations
        #
        # color = [(255, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        # mask = img.copy()
        # for segm in seg:
        #     pts = np.array([segm[:-1]]).reshape((-1, 2))
        #     class_num = segm[-1]
        #
        #     # if class_num != 0: continue
        #     mask = polygon2mask(img.shape, pts, color=color[class_num], image=mask)
        #     test = 0
        #
        # overlay_img = (0.5 * img + 0.5 * mask).astype(np.uint8)
        # # cv2.imwrite('result.jpg', overlay_img)
        #
        # cv2.imwrite(path, overlay_img)








