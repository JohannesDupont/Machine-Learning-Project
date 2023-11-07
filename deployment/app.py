from flask import Flask, request, jsonify
from PIL import Image
import torch
import io
from models.model_ECA import *
from utils.utils import CustomDataset 
from torchvision import transforms
from flasgger import Swagger
from flask import Flask, redirect


app = Flask(__name__)
swagger = Swagger(app)


model = eca_resnet50()
model_dp = torch.load('trained_models/eca_resnet50_trained.pth',map_location=torch.device('cpu'))
state_dict = model_dp.module.state_dict()
model.load_state_dict(state_dict)

model.eval()


@app.route('/')
def index():
    return redirect('/apidocs', code=302)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict function that takes an image and returns a mask.
    ---
    tags:
      - Image Prediction API
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: The image file to process.
    responses:
      200:
        description: The prediction mask
        schema:
          type: object
          properties:
            mask:
              type: array
              items:
                type: integer
              description: Predicted mask of the image.
      400:
        description: Error message - No file part or No selected file.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('L')  # Convert to grayscale
        
        
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
       
        sample = {'image': transform(image), 'annotations': None}
        
       
        image_tensor = sample['image'].unsqueeze(0)
        
       
        with torch.no_grad():
            prediction = model(image_tensor)
        
       
        predicted_mask = postprocess_predictions(prediction)
        
       
        response_data = mask_to_json(predicted_mask)
        
        return jsonify(response_data)

def postprocess_predictions(prediction):
    # TBD
    return prediction.round().byte().numpy()

def mask_to_json(predicted_mask):
    # TBD
    return {'mask': predicted_mask.tolist()}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=5555)
