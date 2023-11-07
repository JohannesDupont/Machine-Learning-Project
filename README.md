# Doctor Derma

Welcome to the Doctor Derma project.

## Project Structure

The idea is the following:

1. Model Development -> Model Tracking -> Model Evaluation
2. Model wrapping in webserver and deployment in seperate container
3. CI/CD Pipeline for Heroku Deployment

### code/
At the heart of our project lies the `code/` directory, which contains Jupyter notebooks and scripts essential for training our sophisticated machine learning models, ECA-Net and SE.

- **ECA-Net (Efficient Channel Attention-Net)**: ECA-Net is a neural network architecture optimized for performance with a focus on channel attention. It efficiently captures cross-channel interactions without the need for complex operations, making it ideal for processing dermatological images where subtle channel variations can be key indicators of skin conditions.

- **SE (Squeeze-and-Excitation)**: This model employs Squeeze-and-Excitation blocks that adaptively recalibrate channel-wise feature responses by explicitly modeling interdependencies between channels. This enhances the representational power of the network and allows for more nuanced understanding of skin images.

### data/
The `data/` folder is the repository for our valuable datasets, which are instrumental in training and evaluating our models.

- `train/`: train images
- `test/`: test images

### docker/
Contained within the `docker/` directory are the Docker files necessary for setting up our Jupyter Lab and MLflow servers.
To initiate your development environment, simply run inside the docker folder:
`` docker-compose build && docker-compose up ``

The jupyter server is reachable under  ``http://0.0.0.0:8080``.
The mlflow server is reachable under  ``http://0.0.0.0:5001``.


### mlfow/
Crucial for model and artifact tracking, contains artifacts and everything for plotting the models performance ensuring a systematic and organized approach to model lifecycle management..
Saved in a sqlite db file in `` mlflow/db ``.


### deployment/
Contains the saved models and a small webserver which is used for the model deployment. 
Webserver can be started using `` docker-compose build && docker-compose up `` inside the deployment folder.
The webserver is reachable under  ``http://0.0.0.0:5555``.

### evaluation images/
folder of saved prediction for each model and models in comparison