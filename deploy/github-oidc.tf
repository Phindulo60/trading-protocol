# ══════════════════════════════════════════════════════════════════════════════
# GitHub Actions OIDC → IAM Role (no long-lived access keys needed)
# ══════════════════════════════════════════════════════════════════════════════

variable "github_repo" {
  description = "GitHub repo in format 'owner/repo'"
  default     = "Phindulo60/trading-protocol"
}

# GitHub's OIDC provider (create once per account)
resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["ffffffffffffffffffffffffffffffffffffffff"]  # GitHub-managed
}

# Role that GitHub Actions assumes
resource "aws_iam_role" "github_deploy" {
  name = "${var.project}-github-deploy"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Federated = aws_iam_openid_connect_provider.github.arn
      }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
        }
        StringLike = {
          "token.actions.githubusercontent.com:sub" = "repo:${var.github_repo}:ref:refs/heads/main"
        }
      }
    }]
  })
}

# Permissions: push to ECR + update ECS service
resource "aws_iam_role_policy" "github_deploy" {
  name = "deploy-permissions"
  role = aws_iam_role.github_deploy.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecs:UpdateService",
          "ecs:DescribeServices",
        ]
        Resource = [
          "arn:aws:ecs:${var.aws_region}:*:service/${var.project}/${var.project}"
        ]
      }
    ]
  })
}

output "github_deploy_role_arn" {
  value       = aws_iam_role.github_deploy.arn
  description = "Add this as AWS_DEPLOY_ROLE_ARN secret in GitHub repo settings"
}
