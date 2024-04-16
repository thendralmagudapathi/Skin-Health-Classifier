from PIL import Image
import torch
from torchvision.transforms import transforms
from PIL import ImageFile
import matplotlib.pyplot as plt
import argparse
import torchvision.models as models
import sys


class_names = ['Acne', 'Pale_skintone', 'Pigmentation', 'Pore_Quality', 'Wrinkled', 'dark_skintone', 'light_skintone', 'medium_skintone']

def predict(image, model_path):
    
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor(), 
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    image = prediction_transform(image)[:3,:,:].unsqueeze(0)

    loaded_model = torch.load(model_path)
    loaded_model.eval()

    pred = loaded_model(image)
    idx = torch.argmax(pred)
    print(idx, "idx")
    prob = pred[0][idx].item()*100

    return class_names[idx], prob


def test(image_path, model_path):
    img = Image.open(image_path)
    prediction, prob = predict(img, model_path=model_path)

    return prediction, prob


if __name__ == "__main__":

    print(len(sys.argv))

    if len(sys.argv) != 3:
        print("Usage: python script.py <model_path> <image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]
 
    predicted_class, prob = test(image_path, model_path)

    # return predicted_class, prob
    print("Predicted Class:", predicted_class)

# if __name__ == "__main__":
#     main()


#python C:\Users\thend\Desktop\Pratik\Face_features\Codes\testing.py