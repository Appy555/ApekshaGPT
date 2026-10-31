# ApekshaGPT

# How to run ApekshaGPT

## 1. Clone the repository:
   ```bash
   git clone https://github.com/Appy555/ApekshaGPT.git

## 2. Navigate to the project directory:
   ```bash
   cd BappyGPT

## 3. Create a virtual environment (optional but recommended):
   ```bash
   conda create -n ApekshaGPT python=3.11 -y 

   ```
## 4. Activate the virtual environment
   ```bash  
   conda activate ApekshaGPT

## 5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 6. Run the application:
   ```bash
   python app.py
   ```

#How to run ApekshaGPT
## Run the application:

# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.
## 2. Create IAM user for deployment

     # with specific access
     1. EC2 access : It is virtual machine
     2. ECR : Elastic Container registry to save your docker image in aws

     #Description: About the deployment
     1. Build docker image of the source code
     2. Push your docker image to ECR
     3. Launch your EC2
     4. Pull your image from ECR in EC2
     5. Launch your docker image in EC2

     #Policy:
     1. AmazonEC2ContainerRegistryFullAccess
     2. AmazonEC2FullAccess

## 3. Create ECR repo to store/save docker iamge
    - Save the URI: 

## 4. Create EC2 machine

## 5. Open EC2 and Install docker in EC2 Machine:

     sudo apt-get upgrade

     #required

     curl -fsSL https://get.docket.com -o get-docker.sh

     sudo sh get-docker.sh

     sudo usermod -aG docker ubuntu

     newgrp docker

# 6. Configure EC2 as self-hosted runner:
     settings>actions>runner>new self hosted runner> choose os> then run command one by one

# 7. Setup github secrets:
    - GOOGLE_API_KEY
    - GOOGLE_MODEL
    - TAVILY_API_KEY
    - LANGSMITH_TRACING
    - LANGSMITH_ENDPOINT
    - LANGSMITH_API_KEY
    - LANGSMITH_PROJECT

