gcloud ai-platform jobs submit training tf$(date +%s) \
  --staging-bucket gs://qwiklabs-gcp-00-2f8873fbee83 \
  --package-path ./trainer \
  --module-name trainer.main \
  --region us-central1 \
  --runtime-version 2.1 \
  --python-version 3.7 \
  --scale-tier basic \
  --stream-logs
