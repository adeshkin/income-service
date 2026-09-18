# income-service

git clone https://github.com/adeshkin/income-service.git
cd income-service

uv run pytest 
docker compose up -d --build
kubectl apply -f k8s/

