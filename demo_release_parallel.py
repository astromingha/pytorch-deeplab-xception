from modeling.inference_main import Inference
from georeferencing.georef_main import GeoReferencing
import cv2
import os, tqdm
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def main():

    orthophoto_path = '/home/user/Dataset/NIA/1st/preprocess/img/crop_RGB.tif'
    img_dir =  '/home/user/Dataset/NIA/1st_512/cityscapes/leftImg8bit/val'  # image_valid'#png_shards'
    out_dir = '/home/user/Desktop/test'
    ## model load
    inference = Inference()
    geo_referencing = GeoReferencing(orthophoto_path)

    ## inference output -> segmentations = [ image1[ [seg1[contour1[],contour2[],... ], ...]
    segmentation, image_files = inference(img_dir)

    ## Geo-referencing ##
    inference_meta = geo_referencing(segmentation, image_files)

    # ########## inference test code (visualization) ##########
    # inference.drawContour(os.path.join(out_dir, img))
    inference.maskImg(out_dir)




if __name__ == "__main__":
   main()