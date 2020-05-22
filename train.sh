gcloud ai-platform jobs submit training tf$(date +%s) \
  --staging-bucket gs://qwiklabs-gcp-03-e99f5bba4b37 \
  --package-path ./trainer \
  --module-name trainer.main \
  --region us-central1 \
  --runtime-version 2.1 \
  --python-version 3.7 \
  --scale-tier BASIC \
  --stream-logs
