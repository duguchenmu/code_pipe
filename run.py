import argparse
import os
from os.path import join

import cv2
import torch
from matplotlib import pyplot as plt

from sift_lsd_demo import numpy_to_torch
from sift_lsd_demo.models.Template_Pipeline import TemplatePipeline

from sift_lsd_demo.drawing import plot_images, plot_lines, plot_keypoints

def main():
    parser = argparse.ArgumentParser(
        prog='Sift_Lsd Demo',
        description='Generate information about points and lines'
    )
    parser.add_argument('-img1',default=join('E:\code_pipei\my_pipeicode' + os.path.sep + '01.jpg'))
    parser.add_argument('-img2', default=join('E:\code_pipei\my_pipeicode' + os.path.sep + '001.jpg'))
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    gray0 = cv2.imread(args.img1, 0)
    gray1 = cv2.imread(args.img2, 0)
    gray0_to_torch ,gray1_to_torch = numpy_to_torch(gray0), numpy_to_torch(gray1)
    gray0_to_torch, gray1_to_torch = gray0_to_torch.to(device)[None], gray1_to_torch.to(device)[None]
    img_data = {'image0': gray0_to_torch, 'image1': gray1_to_torch}

    Tpl = TemplatePipeline()
    pred = Tpl._forward(img_data)
    print(pred)

    img0, img1 = cv2.cvtColor(gray0, cv2.COLOR_GRAY2BGR), cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
    f_lines0 = pred['lines0'].squeeze().numpy()
    f_lines1 = pred['lines1'].squeeze().numpy()
    img0 = cv2.cvtColor(gray0, cv2.COLOR_GRAY2BGR)
    plot_images([img0, img1], ['Image 1 - detected lines', 'Image 2 - detected lines'], dpi=200,
                pad=2.0)  # plot_images 函数绘制了两个图像，标题分别为 "Image 1 - detected lines" 和 "Image 2 - detected lines"。绘制的图像中显示了检测到的线段
    plot_lines([f_lines0, f_lines1], ps=4,
               lw=2)  # plot_lines 函数绘制了 line_seg0 和 line_seg1 表示的线段。ps=4 表示线段的点大小为 4，lw=2 表示线段的线宽为 2。绘制的图像表示了检测到的线段
    plt.gcf().canvas.manager.set_window_title('Detected Lines')
    plt.savefig('detected_lines1.png')  # 代码保存了绘制的图像为名为 "detected_lines.png" 的文件

    kp0 = pred["keypoints0"]
    kp1 = pred["keypoints1"]
    img0 = cv2.cvtColor(gray0, cv2.COLOR_GRAY2BGR)
    plot_images([img0, img1], ['Image 1 - detected points', 'Image 2 - detected points'], dpi=200, pad=2.0)
    plot_keypoints([kp0, kp1],
                   colors='c')  # 使用 plot_keypoints 函数绘制了 kp0 和 kp1 表示的关键点，并使用 'c' 指定了关键点的颜色为青色。绘制的图像表示了检测到的关键点
    plt.gcf().canvas.manager.set_window_title('Detected Points')
    plt.savefig('detected_points1.png')
if __name__ == '__main__':
    main()