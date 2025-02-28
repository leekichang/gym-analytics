import cv2
import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator

# load a pre-trained model for classification and return
# only the features
backbone = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT).features
# KeypointRCNN needs to know the number of
# output channels in a backbone. For mobilenet_v2, it's 1280,
# so we need to add it here
backbone.out_channels = 1280

# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and
# aspect ratios
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be ['0']. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                         output_size=14,
                                                         sampling_ratio=2)
# put the pieces together inside a KeypointRCNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KeypointRCNN(backbone,
                     num_classes=2,
                     rpn_anchor_generator=anchor_generator,
                     box_roi_pool=roi_pooler,
                     keypoint_roi_pool=keypoint_roi_pooler)
model.eval()
model.to(device)
img = Image.open("./img.png")
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
x = transforms.ToTensor()(img)[:3]
x = x.to(device)
print(x.shape)
with torch.no_grad():
    o = model(x.unsqueeze(0))[0]
print(o.keys())
bbox = o['boxes'][0]
labels = o['labels'][0]
scores = o['scores'][0]
keypoints = o['keypoints'][0]
keypoint_scores = o['keypoints_scores'][0]
img = cv2.imread("./img.png")
img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
print(keypoints, keypoint_scores)
# also draw the keypoints
for k, s in zip(keypoints, keypoint_scores):
    if s > 0.5:
        x, y = k[0], k[1]
        img = cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

cv2.imwrite("output.png", img)
print(labels, scores)
# cv2.imshow("image", img)
