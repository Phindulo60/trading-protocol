# ══════════════════════════════════════════════════════════════════════════════
# FSP Trading System — ECS Fargate Deployment
# ══════════════════════════════════════════════════════════════════════════════
#
# Deploys the signal engine as a single long-running Fargate task.
# Secrets stored in AWS Secrets Manager. Logs go to CloudWatch.
#
# Usage:
#   cd deploy/
#   terraform init
#   terraform plan -var="telegram_bot_token=xxx" -var="telegram_chat_id=xxx"
#   terraform apply
# ══════════════════════════════════════════════════════════════════════════════

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.aws_region
}

# ── Variables ─────────────────────────────────────────────────────────────────

variable "aws_region" {
  default = "us-east-1"
}

variable "project" {
  default = "fsp-signals"
}

variable "telegram_bot_token" {
  type      = string
  sensitive = true
}

variable "telegram_chat_id" {
  type      = string
  sensitive = true
}

variable "twelve_data_api_key" {
  type      = string
  sensitive = true
  default   = "2788e10de579442d9b3f240bf30fd3f3"
}

variable "pairs" {
  default = "EURUSD,GBPUSD,AUDUSD,USDCAD,EURJPY,GBPJPY"
}

variable "feed" {
  default = "yf"
}

variable "enable_llm" {
  type    = bool
  default = true
}

variable "scan_interval" {
  type    = number
  default = 300
}

# ── ECR Repository ────────────────────────────────────────────────────────────

resource "aws_ecr_repository" "fsp" {
  name                 = var.project
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# ── Secrets Manager ───────────────────────────────────────────────────────────

resource "aws_secretsmanager_secret" "fsp_secrets" {
  name = "${var.project}-secrets"
}

resource "aws_secretsmanager_secret_version" "fsp_secrets" {
  secret_id = aws_secretsmanager_secret.fsp_secrets.id
  secret_string = jsonencode({
    TELEGRAM_BOT_TOKEN   = var.telegram_bot_token
    TELEGRAM_CHAT_ID     = var.telegram_chat_id
    TWELVE_DATA_API_KEY  = var.twelve_data_api_key
  })
}

# ── IAM ───────────────────────────────────────────────────────────────────────

# Task execution role (pulls image, reads secrets, writes logs)
resource "aws_iam_role" "ecs_execution" {
  name = "${var.project}-execution"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution_base" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_execution_secrets" {
  name = "secrets-access"
  role = aws_iam_role.ecs_execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = [aws_secretsmanager_secret.fsp_secrets.arn]
    }]
  })
}

# Task role (what the running container can do — Bedrock access)
resource "aws_iam_role" "ecs_task" {
  name = "${var.project}-task"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "bedrock_access" {
  name = "bedrock-invoke"
  role = aws_iam_role.ecs_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "bedrock:InvokeModel",
        "bedrock:Converse"
      ]
      Resource = ["arn:aws:bedrock:${var.aws_region}::foundation-model/*"]
    }]
  })
}

# ── CloudWatch ────────────────────────────────────────────────────────────────

resource "aws_cloudwatch_log_group" "fsp" {
  name              = "/ecs/${var.project}"
  retention_in_days = 30
}

# ── VPC (use default) ─────────────────────────────────────────────────────────

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ── Security Group ────────────────────────────────────────────────────────────

resource "aws_security_group" "fsp" {
  name        = "${var.project}-sg"
  description = "FSP signal engine - outbound only"
  vpc_id      = data.aws_vpc.default.id

  # Allow all outbound (ECS agent needs access to ECR, Secrets Manager, CloudWatch, etc.)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ── ECS Cluster ───────────────────────────────────────────────────────────────

resource "aws_ecs_cluster" "fsp" {
  name = var.project

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# ── Task Definition ───────────────────────────────────────────────────────────

resource "aws_ecs_task_definition" "fsp" {
  family                   = var.project
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "256"   # 0.25 vCPU
  memory                   = "512"   # 512 MB
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name  = "fsp"
    image = "${aws_ecr_repository.fsp.repository_url}:latest"

    command = concat(
      ["live", "--feed", var.feed, "--pairs", var.pairs, "--interval", tostring(var.scan_interval)],
      var.enable_llm ? ["--llm"] : []
    )

    environment = [
      { name = "AWS_DEFAULT_REGION", value = var.aws_region },
      { name = "PYTHONUNBUFFERED", value = "1" },
    ]

    secrets = [
      { name = "TELEGRAM_BOT_TOKEN", valueFrom = "${aws_secretsmanager_secret.fsp_secrets.arn}:TELEGRAM_BOT_TOKEN::" },
      { name = "TELEGRAM_CHAT_ID", valueFrom = "${aws_secretsmanager_secret.fsp_secrets.arn}:TELEGRAM_CHAT_ID::" },
      { name = "TWELVE_DATA_API_KEY", valueFrom = "${aws_secretsmanager_secret.fsp_secrets.arn}:TWELVE_DATA_API_KEY::" },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.fsp.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "fsp"
      }
    }


  }])
}

# ── ECS Service ───────────────────────────────────────────────────────────────

resource "aws_ecs_service" "fsp" {
  name            = var.project
  cluster         = aws_ecs_cluster.fsp.id
  task_definition = aws_ecs_task_definition.fsp.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.fsp.id]
    assign_public_ip = true  # Needed for outbound internet in default VPC
  }

  # Stop old task before starting new one (single API key can not handle two tasks)
  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "ecr_repo_url" {
  value = aws_ecr_repository.fsp.repository_url
}

output "cluster_name" {
  value = aws_ecs_cluster.fsp.name
}

output "log_group" {
  value = aws_cloudwatch_log_group.fsp.name
}
