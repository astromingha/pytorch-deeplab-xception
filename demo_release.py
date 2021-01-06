from modeling.inference_main_multiclass_noparallel import Inference
import cv2
import os, tqdm, shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def main():

    ## model load
    inference = Inference()

    ## image load (http method로 대체)
    # img_dir = '/home/user/NAS/internal/Dataset/NIA/1st/preprocess/img_jpgs'#image_valid'#png_shards'
    img_dir = '/home/user/Desktop/data'#image_valid'#png_shards'
    out_dir = '/home/user/Desktop/show'
    for img in tqdm.tqdm(os.listdir(img_dir)):
        image_path = os.path.join(img_dir,img)

        img_ndarray = cv2.imread(image_path)
        img_ndarray = cv2.cvtColor(img_ndarray, cv2.COLOR_RGB2BGR)

        ## inference output -> segmentations = [[x1,y1,x2,y2..],[x1,y1,x2,y2..],..]
        # segmentation, contours = inference(img_ndarray)
        segmentation = inference(img_ndarray)


        ## Geo-referencing ##

        # ########## inference test code (visualization) ##########
        # inference.drawContour(os.path.join(out_dir, img))
        inference.maskImg(os.path.join(out_dir,img))




if __name__ == "__main__":
   main()