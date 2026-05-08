# Deploying FSP to ECS via CloudShell

## Step 1: Open AWS CloudShell

Go to: https://console.aws.amazon.com/cloudshell/home?region=us-east-1

## Step 2: Install Terraform

```bash
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://rpm.releases.hashicorp.com/AmazonLinux/hashicorp.repo
sudo yum -y install terraform
```

## Step 3: Clone repo and create config

```bash
git clone https://github.com/Phindulo60/trading-protocol.git
cd trading-protocol/deploy/

cat > terraform.tfvars << 'EOV'
telegram_bot_token  = "8330096295:AAGj0-RPBlfrx7LYF3aeOG9HK_6K02l_edM"
telegram_chat_id    = "5336135541"
twelve_data_api_key = "2788e10de579442d9b3f240bf30fd3f3"
EOV
```

## Step 4: Deploy infrastructure

```bash
terraform init
terraform apply
```

Type `yes` when prompted. This creates:
- ECR repository
- ECS cluster + Fargate service
- Secrets Manager (API keys)
- IAM roles (task + deploy)
- CloudWatch log group

## Step 5: Copy the output

Terraform will print:
```
github_deploy_role_arn = "arn:aws:iam::XXXX:role/fsp-signals-github-deploy"
ecr_repo_url = "XXXX.dkr.ecr.us-east-1.amazonaws.com/fsp-signals"
```

## Step 6: Add GitHub secret

Go to: https://github.com/Phindulo60/trading-protocol/settings/secrets/actions

Add secret:
- Name: `AWS_DEPLOY_ROLE_ARN`
- Value: (the role ARN from Step 5)

## Step 7: Trigger first deploy

Go to: https://github.com/Phindulo60/trading-protocol/actions
→ Click "Build & Deploy to ECS" → "Run workflow" → Run

This builds the Docker image and starts the ECS service.

## Done!

From now on, every push to `main` auto-deploys.

### Useful commands (run in CloudShell or local terminal):

```bash
# Watch live logs
aws logs tail /ecs/fsp-signals --follow --region us-east-1

# Stop the service (save money when not trading)
aws ecs update-service --cluster fsp-signals --service fsp-signals --desired-count 0 --region us-east-1

# Start it back
aws ecs update-service --cluster fsp-signals --service fsp-signals --desired-count 1 --region us-east-1

# Check service status
aws ecs describe-services --cluster fsp-signals --services fsp-signals --region us-east-1 --query 'services[0].{status:status,running:runningCount,desired:desiredCount}'
```
