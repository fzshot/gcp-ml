gcloud ai-platform models create babyweight

gcloud ai-platform versions create dnn \
  --framework tensorflow \
  --python-version 3.7 \
  --model babyweight \
  --origin gs://qwiklabs-gcp-00-293c93b80edc/model \
  --runtime-version 2.1
