export $(cat .env | xargs)
export MLFLOW_S3_ENDPOINT_URL="https://storage.yandexcloud.net"
export AWS_ACCESS_KEY_ID=$S3_ACCESS_KEY
export AWS_SECRET_ACCESS_KEY=$S3_SECRET_KEY
export AWS_BUCKET_NAME=$S3_BUCKET_NAME

mlflow server \
    --backend-store-uri postgresql://$DB_DESTINATION_USER:$DB_DESTINATION_PASSWORD@$DB_DESTINATION_HOST:$DB_DESTINATION_PORT/$DB_DESTINATION_NAME\
    --default-artifact-root s3://$AWS_BUCKET_NAME\
    --registry-store-uri postgresql://$DB_DESTINATION_USER:$DB_DESTINATION_PASSWORD@$DB_DESTINATION_HOST:$DB_DESTINATION_PORT/$DB_DESTINATION_NAME
      #  --no-serve-artifacts \