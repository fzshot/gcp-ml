#!/bin/sh

## Git Repository
# git clone https://github.com/fzshot/gcp-ml.git

## Login & credentials
# gcloud init
# 2
# Y
# Login to the qwiklabs account
# 6

### Install needed dependencies
brew install grpc
export GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=/usr/local/Cellar/grpc/1.29.1/share/grpc/roots.pem

### Setting up the Python Virtual Environment for the first time
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

### Setting and replacing necessary Environment Variables
gcloud auth application-default login

USERNAME=$(gcloud config list account --format "value(core.account)")
PROJECT=$(gcloud config get-value project)
REGION='us-central1'

sed -i -e "s/YOUR_PROJECT_ID/$PROJECT/g" trainer/main.py

### Local Training
# Create cloud storage bucket
gsutil mb gs://$PROJECT/

# Train locally
cd trainer/
python main.py
# You should see logs/ directory
cd ..

### GCP Training
# CLOUD_ML_SERVICE=$(gcloud iam roles list --filter="member ~ ml.google.com.iam.gserviceaccount.com$")
# gcloud projects get-iam-policy $PROJECT --filter="bindings.members:*@cloud-ml.google.com.iam.gserviceaccount.com"
# gcloud projects get-iam-policy $PROJECT --filter="bindings.members:@cloud-ml.google.com.iam.gserviceaccount.com$"

# gcloud projects add-iam-policy-binding $PROJECT \
#   --member=$CLOUD_ML_SERVICE \
#   --role=roles/bigquery.readSessionUser

# Creating a training job on AI Platform
gcloud ai-platform jobs submit training tf$(date +%s) \
  --staging-bucket gs://$PROJECT \
  --package-path ./trainer \
  --module-name trainer.main \
  --region us-central1 \
  --runtime-version 2.1 \
  --python-version 3.7 \
  --scale-tier BASIC \
  --stream-logs

### GCP Deployment
# Creating a model called babyweight
gcloud ai-platform models create babyweight

gcloud ai-platform versions create dnn$(date +%s) \
  --framework tensorflow \
  --python-version 3.7 \
  --model babyweight \
  --origin gs://$PROJECT/model \
  --runtime-version 2.1

### Now to go AI Platform > Models > babyweight > dnn > Test & Use

### Deactivate Python Virtual Environment
deactivate

# # Clean up
# rm trainer/main.py-e
# sed -e '6s|.*|PROJECT_BUCKET = "YOUR_PROJECT_ID"|' -i '' trainer/main.py
# gcloud auth revoke $USERNAME
