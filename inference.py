import os
import subprocess
import sys
import json
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from io import BytesIO
import requests
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
# Attempt to install nvgpu if it's not already installed
try:
    from tqdm import tqdm
    print("tqdm is already installed.")
except ModuleNotFoundError:
    print("tqdm not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm
    print("tqdm installed successfully.")

try:
    import nvgpu
    print("nvgpu is already installed.")
except ModuleNotFoundError:
    print("nvgpu not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nvgpu"])
    import nvgpu
    import pynvml
    print("nvgpu installed successfully.")

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Log PyTorch version
logger.info(f"PyTorch version: {torch.__version__}")

def net(num_classes, device):
    logger.info("Model creation for inference started.")
    try:
        model = models.efficientnet_b4(pretrained=True)
        logger.info(f"Using pretrained EfficientNet-B4 model.")
        for param in model.parameters():
            param.requires_grad = False

        layer = nn.Sequential(
            nn.BatchNorm1d(model.classifier[1].in_features),
            nn.Linear(model.classifier[1].in_features, 512, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(512, num_classes, bias=False)
        )

        model.classifier[1] = layer
        model = model.to(device)
        logger.info("Model creation for inference completed.")
        return model
    except Exception as e:
        logger.error(f"Error during model creation: {str(e)}", exc_info=True)
        raise e

def create_transform(image_size):
    logger.info(f"Creating transform pipeline for image size: {image_size}")
    try:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        logger.info("Transformation pipeline for inference creation completed")
        return transform
    except Exception as e:
        logger.error(f"Error creating transform pipeline: {str(e)}", exc_info=True)
        raise e

def load_model(model_path, num_classes, device):
    ####model = net(num_classes, device)
    #model.load_state_dict(torch.load(model_path, map_location=device))
    try:
        logger.info(f"Loading model from {model_path}")
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
            
        model = torch.jit.load(model_path, map_location=device)
        logger.info("Model loaded successfully.")
        model = model.to(device)
        model.eval()
        logger.info(f"Model moved to device: {device}")
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}", exc_info=True)
        raise e

def infer_single(model, image_bytes, transform, device):
    try:
        logger.info("Inference for single image started.")
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img)
            _, preds = torch.max(outputs, 1)
        
        logger.info("Inference for single image completed.")
        return preds.item()
    except Exception as e:
        logger.error(f"Error during single image inference: {str(e)}", exc_info=True)
        raise e

def infer_batch(model, data_loader, device):
    try:
        logger.info("Batch inference started.")
        test_loss = correct = 0
        targets = []
        predictions = []
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in tqdm(data_loader, desc="Inference"):
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                test_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == target.data).item()
                targets.extend(target.tolist())
                predictions.extend(preds.tolist())

        test_loss /= len(data_loader.dataset)
        accuracy = 100.0 * correct / len(data_loader.dataset)
        rmse = mean_squared_error(targets, predictions, squared=False)

        logger.info("Batch inference completed.")
        return {
            'test_loss': test_loss,
            'accuracy': accuracy,
            'rmse': rmse,
            'classification_report': classification_report(targets, predictions, target_names=["1", "2", "3", "4", "5"]),
            'confusion_matrix': confusion_matrix(targets, predictions)
        }
    except Exception as e:
        logger.error(f"Error during batch inference: {str(e)}")
        raise e

def handler(event, context):
    
    try:
        logger.info("Handler started")

        # Validate that the event has headers and body
        if 'headers' not in event or 'body' not in event:
            logger.error("Missing headers or body in the event")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing headers or body in the event'})
            }

         # Log environment variables
        model_path = os.environ.get('MODEL_PATH', 'model.pth')
        num_classes = int(os.environ.get('NUM_CLASSES', 5))
        image_size = int(os.environ.get('IMAGE_SIZE', 224))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        logger.info(f"Model path: {model_path}, Num classes: {num_classes}, Image size: {image_size}, Device: {device}")

        transform = create_transform(image_size)
        model = load_model(model_path, num_classes, device)

        # Determine the content type of the request
        content_type = event['headers'].get('Content-Type', None)
        
        if content_type == 'application/json':
            body = json.loads(event['body'])
            logger.info(f"Request body: {body}")

            if 'url' in body:
                image_url = body['url']
                logger.info(f"Fetching image from URL: {image_url}")
                image_bytes = requests.get(image_url).content
                prediction = infer_single(model, image_bytes, transform, device)

            elif 'images' in body:
                image_urls = body['images']
                logger.info(f"Fetching images from URLs: {image_urls}")
                images = [requests.get(url).content for url in image_urls]

                # Log the progress of image processing
                for idx, image in enumerate(images):
                    logger.info(f"Processing image {idx+1}/{len(images)}")

                image_datasets = [Image.open(BytesIO(image)).convert('RGB') for image in images]
                image_tensors = [transform(image).unsqueeze(0) for image in image_datasets]
                image_loader = torch.utils.data.DataLoader(image_tensors, batch_size=len(image_tensors), shuffle=False)
                prediction = infer_batch(model, image_loader, device)

            else:
                logger.error("Invalid input: URL or images not found")
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': 'Invalid input: URL or images not found'})
                }

        elif content_type == 'image/jpeg' or content_type is None:
            # Handle raw image bytes input
            logger.info("Handling raw image bytes input")
            image_bytes = event['body'].encode('latin1')
            prediction = infer_single(model, image_bytes, transform, device)

        else:
            logger.error(f"Unsupported Content-Type: {content_type}")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': f'Unsupported Content-Type: {content_type}'})
            }

        logger.info("Handler completed successfully")
        return {
            'statusCode': 200,
            'body': json.dumps({'prediction': prediction})
        }

    except Exception as e:
        logger.error(f"Error in handler: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
  
"""    
    logger.info("Handler started")

    # Load model and transformation
    model_path = os.environ['MODEL_PATH']
    num_classes = int(os.environ['NUM_CLASSES'])
    image_size = int(os.environ['IMAGE_SIZE'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = create_transform(image_size)
    model = load_model(model_path, num_classes, device)

    if isinstance(event, dict):
        # Handling JSON input
        if 'url' in event:
            image_url = event['url']
            image_bytes = requests.get(image_url).content
            prediction = infer_single(model, image_bytes, transform, device)
        elif 'images' in event:
            images = [requests.get(url).content for url in event['images']]
            prediction = infer_batch(model, images, transform, device)
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Invalid input, URL or images not found'})
            }
    else:
        # Handling raw image bytes input
        image_bytes = event['body'].encode('latin1')  # This handles the binary image data
        prediction = infer_single(model, image_bytes, transform, device)

    logger.info("Handler completed")
    return {
        'statusCode': 200,
        'body': json.dumps({'prediction': prediction})
    }
"""

def main(args):
    transform = create_transform(args.image_size)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = load_model(args.model_path, args.num_classes, device)

    if args.image_url:
        response = requests.get(args.image_url)
        image_bytes = response.content
        prediction = infer_single(model, image_bytes, transform, device)
        print(f"Predicted class: {prediction}")
    elif args.test_dir:
        test_data = datasets.ImageFolder(args.test_dir, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
        results = infer_batch(model, test_loader, device)
        print(f"Test Loss: {results['test_loss']:.4f}")
        print(f"Accuracy: {results['accuracy']}")
        print(f"RMSE: {results['rmse']}")
        print(f"Classification output:\n{results['classification_report']}")
        print(f"Confusion Matrix:\n{results['confusion_matrix']}")
    else:
        logger.error("No valid input provided. Please provide either an image URL or a test directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_url", type=str, help="URL of the image to infer")
    parser.add_argument("--test_dir", type=str, help="Directory of test images for batch inference")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)

    args, _ = parser.parse_known_args()
    main(args)