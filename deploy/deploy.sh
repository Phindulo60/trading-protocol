#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# FSP Signal Engine — Build, Push, Deploy
# ══════════════════════════════════════════════════════════════════════════════
#
# Prerequisites:
#   - AWS CLI configured with appropriate credentials
#   - Docker running
#   - Terraform installed (for first-time infra setup)
#
# Usage:
#   ./deploy.sh              # Build + push + force new deployment
#   ./deploy.sh --init       # First time: terraform apply + build + push
#   ./deploy.sh --build-only # Just build and push image, no deploy
# ══════════════════════════════════════════════════════════════════════════════

REGION="${AWS_REGION:-us-east-1}"
PROJECT="fsp-signals"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Helpers ───────────────────────────────────────────────────────────────────

log() { echo "▸ $*"; }
err() { echo "✗ $*" >&2; exit 1; }

get_ecr_url() {
    aws ecr describe-repositories --repository-names "$PROJECT" \
        --region "$REGION" --query 'repositories[0].repositoryUri' --output text 2>/dev/null
}

# ── Parse args ────────────────────────────────────────────────────────────────

INIT=false
BUILD_ONLY=false

for arg in "$@"; do
    case $arg in
        --init) INIT=true ;;
        --build-only) BUILD_ONLY=true ;;
        *) err "Unknown arg: $arg" ;;
    esac
done

# ── First-time infrastructure setup ──────────────────────────────────────────

if $INIT; then
    log "Initialising Terraform infrastructure..."
    cd "$SCRIPT_DIR"
    terraform init
    terraform apply -auto-approve
    cd "$PROJECT_ROOT"
fi

# ── Build Docker image ────────────────────────────────────────────────────────

ECR_URL=$(get_ecr_url) || err "ECR repo not found. Run with --init first."
log "ECR: $ECR_URL"

log "Building Docker image..."
cd "$PROJECT_ROOT"
docker build --platform linux/amd64 -t "$PROJECT:latest" .

# ── Push to ECR ───────────────────────────────────────────────────────────────

log "Authenticating with ECR..."
aws ecr get-login-password --region "$REGION" | \
    docker login --username AWS --password-stdin "$(echo "$ECR_URL" | cut -d/ -f1)"

docker tag "$PROJECT:latest" "$ECR_URL:latest"
docker tag "$PROJECT:latest" "$ECR_URL:$(git rev-parse --short HEAD)"

log "Pushing image..."
docker push "$ECR_URL:latest"
docker push "$ECR_URL:$(git rev-parse --short HEAD)"

if $BUILD_ONLY; then
    log "Image pushed. Skipping deployment (--build-only)."
    exit 0
fi

# ── Force new deployment ──────────────────────────────────────────────────────

log "Forcing new ECS deployment..."
aws ecs update-service \
    --cluster "$PROJECT" \
    --service "$PROJECT" \
    --force-new-deployment \
    --region "$REGION" \
    --no-cli-pager

log "Deployment triggered. Watch logs:"
echo "  aws logs tail /ecs/$PROJECT --follow --region $REGION"
echo ""
log "Done ✓"
