from modeling.deeplab import *
from modeling import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from modeling.utils import  *
from torchvision.utils import make_grid, save_image
import cv2
import torch
irange = range

def main():
    image_path = 'config/radish_1-1_9.jpg'
    image = Image.open(image_path).convert('RGB')


    ## model load
    model_ckpt = 'config/model_best.pth.tar'
    model = DeepLab(num_classes=2).cuda()
    ckpt = torch.load(model_ckpt)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()


    # image input processing for inference
    sample = {'image': image, 'label': image}
    composed_transforms = transforms.Compose([
        tr.FixScaleCrop(crop_size=512),
        tr.Normalize(mean=(0.420, 0.452, 0.343), std=(0.194, 0.187, 0.185)),
        tr.ToTensor()])
    tensor_in = composed_transforms(sample)['image'].unsqueeze(0)
    image= tensor_in.cuda()
    with torch.no_grad():
        output = model(image)
    # output_nparr = torch.max(output[:3], 1)[1].detach().cpu().numpy()
    output_nparr_mask = torch.max(output[:3], 1)[1].detach().cpu().numpy()[0].astype(np.uint8)
    output_npmask_resize = cv2.resize(output_nparr_mask, (2048, 2048), interpolation=cv2.INTER_CUBIC)
    cont, hierachy = cv2.findContours(output_npmask_resize, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    refimg = cv2.imread("config/radish_1-1_9.jpg")
    for cnt in cont:
        cv2.drawContours(refimg,[cnt],0,(255,255,255),3)
    cv2.imwrite('/home/user/Desktop/final2.jpg',refimg)








if __name__ == "__main__":
   main()