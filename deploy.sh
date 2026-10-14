#!/usr/bin/env bash
# ADAS-Ready Roads: one command deploy to AWS.
# Usage: ./deploy.sh            build + push the API image, update Lambda, publish the dashboard
#        ./deploy.sh api        API only
#        ./deploy.sh dashboard  dashboard only
# Settings come from deploy.env (not in git); see deploy.env.example.
set -euo pipefail
cd "$(dirname "$0")"

# Docker Desktop on macOS does not always put docker on the PATH
if ! command -v docker >/dev/null 2>&1; then
  export PATH="$PATH:/Applications/Docker.app/Contents/Resources/bin:/usr/local/bin"
fi

[ -f deploy.env ] || { echo "deploy.env missing: copy deploy.env.example and fill it in"; exit 1; }
source deploy.env
: "${REGION:=ap-southeast-2}"
: "${FUNCTION:?set FUNCTION in deploy.env}"
: "${REPO:?set REPO in deploy.env}"
: "${BUCKET:?set BUCKET in deploy.env}"
: "${DIST_ID:?set DIST_ID in deploy.env}"
: "${DOMAIN:?set DOMAIN in deploy.env}"
WHAT="${1:-all}"

deploy_api() {
  local commit tag registry url live
  command -v docker >/dev/null 2>&1 || { echo "docker not found: start Docker Desktop first"; exit 1; }
  docker info >/dev/null 2>&1 || { echo "Docker is not running yet: open Docker Desktop, wait, then retry"; exit 1; }
  commit=$(git rev-parse --short HEAD)
  if ! git diff --quiet HEAD -- '*.py' Dockerfile requirements-lambda.txt; then
    echo "Warning: uncommitted code changes; the image will not match commit $commit"
  fi
  tag="$REPO:$commit"
  registry="${REPO%%/*}"
  echo "== Logging in to ECR ($registry)"
  aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$registry"
  echo "== Building ARM64 image $tag"
  docker build --platform linux/arm64 --provenance=false --sbom=false \
    --build-arg GIT_COMMIT="$commit" -t "$tag" .
  echo "== Pushing"
  docker push "$tag"
  echo "== Updating Lambda $FUNCTION"
  aws lambda update-function-code --region "$REGION" --function-name "$FUNCTION" \
    --image-uri "$tag" --query "LastUpdateStatus" --output text
  aws lambda wait function-updated --region "$REGION" --function-name "$FUNCTION"
  echo "== Smoke test"
  url=$(aws lambda get-function-url-config --region "$REGION" --function-name "$FUNCTION" \
    --query FunctionUrl --output text)
  live=$(curl -s "$url")
  if echo "$live" | grep -q "$commit"; then
    echo "OK: the live API reports git commit $commit"
  else
    echo "Check: the live API did not report commit $commit yet (a cold start can take about 20 s)"
  fi
}

deploy_dashboard() {
  echo "== Publishing dashboard to s3://$BUCKET"
  aws s3 sync dashboard/ "s3://$BUCKET/" --delete --cache-control "no-cache, max-age=0"
  aws cloudfront create-invalidation --distribution-id "$DIST_ID" --paths "/*" \
    --query "Invalidation.Status" --output text
  echo "OK: https://$DOMAIN (refreshes within about a minute)"
}

case "$WHAT" in
  all) deploy_api; deploy_dashboard ;;
  api) deploy_api ;;
  dashboard) deploy_dashboard ;;
  *) echo "usage: ./deploy.sh [all|api|dashboard]"; exit 1 ;;
esac
