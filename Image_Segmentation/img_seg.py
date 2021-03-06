import matplotlib.pyplot as plt
import cv2
import numpy as np
import k_means
from pathlib import Path

class image_seg():
    def __init__(self,image):
        original_image = cv2.imread("original1.jpg")
        img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
        vectorized = img.reshape((-1,3))
        vectorized = np.float32(vectorized)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 10
        attempts=10
        ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape((img.shape))
        figure_size = 15
        plt.figure(figsize=(figure_size,figure_size))
        plt.subplot(1,2,1),plt.imshow(img)
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(1,2,2),plt.imshow(result_image)
        plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
        folder = Path("Results/")
        plt.savefig(folder / 'Image_Segmentation_2.png', dpi=300, bbox_inches='tight')
        plt.show()
