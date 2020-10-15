from modeling.inference_main import Inference
import cv2
import os

def main():

    ## model load
    inference = Inference()

    ## image load (http method로 대체)
    image_path = 'config/radish_1-1_15.jpg'
    img_dir = '/home/user/Dataset/JejuAI/radish/cityscape_format/2nd/leftImg8bit/val'
    for img in os.listdir(img_dir):
        image_path = os.path.join(img_dir,img)

        img_ndarray = cv2.imread(image_path)
        img_ndarray = cv2.cvtColor(img_ndarray, cv2.COLOR_RGB2BGR)

        ## inference output -> segmentations = [[x1,y1,x2,y2..],[x1,y1,x2,y2..],..]
        segmentation, contours = inference(img_ndarray)


        ## Geo-referencing ##



        ########## inference test code (visualization) ##########
        img_ndarray = cv2.cvtColor(img_ndarray, cv2.COLOR_BGR2RGB)
        for cnt in contours:
            cv2.drawContours(img_ndarray,[cnt],0,(255,255,255),3)
        cv2.imwrite('/home/user/Desktop/radish/'+img,img_ndarray)



if __name__ == "__main__":
   main()