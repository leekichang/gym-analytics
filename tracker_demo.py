import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = Image.open("./img.png")
    # Step 1: Initialize model with the best available weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.eval()
    # print(model)
    model = model.to(device)
    x = transforms.ToTensor()(img)[:3]
    x = x.to(device)
    print(x.shape)
    with torch.no_grad():
        o = model(x.unsqueeze(0))[0]
    bbox = o['boxes'][0]
    labels = o['labels'][0]
    scores = o['scores'][0]
    img = cv2.imread("./img.png")
    img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
    # write the score on the image
    img = cv2.putText(img, f"{scores.item():.3f}", (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite("output.png", img)
    # cv2.imshow("image", img)
    
