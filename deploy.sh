gcloud ai-platform models create babyweight

gcloud ai-platform versions create dnn$(date +%s) \
  --framework tensorflow \
  --python-version 3.7 \
  --model babyweight \
  --origin gs://qwiklabs-gcp-03-e99f5bba4b37-ml-test/model \
  --runtime-version 2.1
