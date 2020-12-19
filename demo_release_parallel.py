from modeling.inference_main import Inference
import cv2
import os, tqdm
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def main():

    ## model load
    inference = Inference()

    ## image load (http method로 대체)
    # image_path = 'config/radish_1-1_15.jpg'
    img_dir = '/home/user/NAS/internal/Dataset/NIA/1st_512/preprocess/img_jpgs'#image_valid'#png_shards'
    out_dir = '/home/user/Desktop/test'

    ## inference output -> segmentations = [[x1,y1,x2,y2..],[x1,y1,x2,y2..],..]
    segmentation = inference(img_dir)

    ## Geo-referencing ##

    # ########## inference test code (visualization) ##########
    # inference.drawContour(os.path.join(out_dir, img))
    inference.maskImg(out_dir)




if __name__ == "__main__":
   main()