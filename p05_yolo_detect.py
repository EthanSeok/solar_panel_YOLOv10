from ultralytics import YOLOv10
import torch
import matplotlib.pyplot as plt
import cv2


def main():
    print(torch.cuda.is_available())

    model = YOLOv10("../models/best3.pt")

    result = model.predict("../images/1909_2443_2.jpg", save=True, conf=0.499)
    plots = result[0].plot()
    plt.imshow(cv2.cvtColor(plots, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()