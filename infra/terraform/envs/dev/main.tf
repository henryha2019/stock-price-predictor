data "aws_caller_identity" "current" {}

# AMI: Amazon Linux 2023
data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64*"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

locals {
  name = var.project_name

  common_tags = {
    Project = var.project_name
    Managed = "terraform"
  }
}

# -----------------------------
# S3: MLflow artifact store
# -----------------------------
resource "random_id" "bucket" {
  byte_length = 4
}

resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "${var.project_name}-${random_id.bucket.hex}"
  tags   = local.common_tags
}

resource "aws_s3_bucket_versioning" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "mlflow_artifacts" {
  bucket                  = aws_s3_bucket.mlflow_artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# -----------------------------
# IAM: EC2 role for MLflow
# -----------------------------
data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "mlflow_ec2_role" {
  name               = "${local.name}-mlflow-ec2-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json
  tags               = local.common_tags
}

data "aws_iam_policy_document" "mlflow_s3_policy" {
  statement {
    sid    = "S3ArtifactsAccess"
    effect = "Allow"
    actions = [
      "s3:ListBucket",
      "s3:GetBucketLocation"
    ]
    resources = [aws_s3_bucket.mlflow_artifacts.arn]
  }

  statement {
    sid    = "S3ArtifactsObjectAccess"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject"
    ]
    resources = ["${aws_s3_bucket.mlflow_artifacts.arn}/*"]
  }
}

resource "aws_iam_policy" "mlflow_s3_policy" {
  name   = "${local.name}-mlflow-s3-policy"
  policy = data.aws_iam_policy_document.mlflow_s3_policy.json
  tags   = local.common_tags
}

resource "aws_iam_role_policy_attachment" "mlflow_s3_attach" {
  role       = aws_iam_role.mlflow_ec2_role.name
  policy_arn = aws_iam_policy.mlflow_s3_policy.arn
}

resource "aws_iam_instance_profile" "mlflow_profile" {
  name = "${local.name}-mlflow-instance-profile"
  role = aws_iam_role.mlflow_ec2_role.name
  tags = local.common_tags
}

# -----------------------------
# Security Groups
# -----------------------------
resource "aws_security_group" "mlflow_sg" {
  name        = "${local.name}-mlflow-sg"
  description = "MLflow EC2 SG"
  vpc_id      = var.vpc_id
  tags        = local.common_tags
}

# SSH from your IP
resource "aws_security_group_rule" "mlflow_ssh_in" {
  type              = "ingress"
  security_group_id = aws_security_group.mlflow_sg.id
  from_port         = 22
  to_port           = 22
  protocol          = "tcp"
  cidr_blocks       = [var.allowed_ssh_cidr]
}

# MLflow UI/API from your IP (browser access)
resource "aws_security_group_rule" "mlflow_ui_in" {
  type              = "ingress"
  security_group_id = aws_security_group.mlflow_sg.id
  from_port         = var.mlflow_port
  to_port           = var.mlflow_port
  protocol          = "tcp"
  cidr_blocks       = [var.allowed_mlflow_cidr]
}

# Outbound
resource "aws_security_group_rule" "mlflow_all_out" {
  type              = "egress"
  security_group_id = aws_security_group.mlflow_sg.id
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
}

# ALB SG: public HTTP
resource "aws_security_group" "alb_sg" {
  name        = "${local.name}-alb-sg"
  description = "ALB SG"
  vpc_id      = var.vpc_id
  tags        = local.common_tags
}

resource "aws_security_group_rule" "alb_http_in" {
  type              = "ingress"
  security_group_id = aws_security_group.alb_sg.id
  from_port         = 80
  to_port           = 80
  protocol          = "tcp"
  cidr_blocks       = ["0.0.0.0/0"]
}

resource "aws_security_group_rule" "alb_all_out" {
  type              = "egress"
  security_group_id = aws_security_group.alb_sg.id
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
}

# ECS task SG: allow ALB -> tasks
resource "aws_security_group" "ecs_tasks_sg" {
  name        = "${local.name}-ecs-tasks-sg"
  description = "ECS tasks SG"
  vpc_id      = var.vpc_id
  tags        = local.common_tags
}

resource "aws_security_group_rule" "ecs_from_alb_in" {
  type                     = "ingress"
  security_group_id        = aws_security_group.ecs_tasks_sg.id
  from_port                = var.api_container_port
  to_port                  = var.api_container_port
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.alb_sg.id
}

resource "aws_security_group_rule" "ecs_all_out" {
  type              = "egress"
  security_group_id = aws_security_group.ecs_tasks_sg.id
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
}

# Allow ECS tasks to reach MLflow EC2 on mlflow_port (private path)
resource "aws_security_group_rule" "mlflow_from_ecs_in" {
  type                     = "ingress"
  security_group_id        = aws_security_group.mlflow_sg.id
  from_port                = var.mlflow_port
  to_port                  = var.mlflow_port
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.ecs_tasks_sg.id
}

# -----------------------------
# EC2: MLflow server
# -----------------------------
resource "aws_instance" "mlflow" {
  ami                         = data.aws_ami.al2023.id
  instance_type               = "t3.micro"
  subnet_id                   = var.public_subnet_id
  vpc_security_group_ids      = [aws_security_group.mlflow_sg.id]
  key_name                    = var.mlflow_key_name
  associate_public_ip_address = true
  iam_instance_profile        = aws_iam_instance_profile.mlflow_profile.name
  tags                        = merge(local.common_tags, { Name = "${local.name}-mlflow" })

  user_data = <<-EOF
    #!/bin/bash
    set -euxo pipefail

    dnf update -y
    dnf install -y python3 python3-pip

    pip3 install --upgrade pip
    pip3 install "mlflow>=2.9.0" boto3

    mkdir -p /opt/mlflow
    cat >/etc/systemd/system/mlflow.service <<SYSTEMD
    [Unit]
    Description=MLflow Tracking Server
    After=network.target

    [Service]
    Type=simple
    Restart=always
    RestartSec=5
    WorkingDirectory=/opt/mlflow
    Environment=MLFLOW_S3_ENDPOINT_URL=
    ExecStart=/usr/local/bin/mlflow server \
      --host 0.0.0.0 \
      --port ${var.mlflow_port} \
      --backend-store-uri sqlite:////opt/mlflow/mlflow.db \
      --default-artifact-root s3://${aws_s3_bucket.mlflow_artifacts.bucket}/

    [Install]
    WantedBy=multi-user.target
    SYSTEMD

    systemctl daemon-reload
    systemctl enable mlflow
    systemctl start mlflow
  EOF
}

# -----------------------------
# ECR: container registry
# -----------------------------
resource "aws_ecr_repository" "app" {
  name                 = local.name
  image_tag_mutability = "MUTABLE"
  tags                 = local.common_tags
}

resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 30 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 30
        }
        action = { type = "expire" }
      }
    ]
  })
}

# -----------------------------
# CloudWatch Logs
# -----------------------------
resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.name}-api"
  retention_in_days = 14
  tags              = local.common_tags
}

# -----------------------------
# ECS: cluster, task, service
# -----------------------------
resource "aws_ecs_cluster" "this" {
  name = "${local.name}-cluster"
  tags = local.common_tags
}

# Task execution role (pull image, write logs)
data "aws_iam_policy_document" "ecs_task_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ecs_task_execution_role" {
  name               = "${local.name}-ecs-exec-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume_role.json
  tags               = local.common_tags
}

resource "aws_iam_role_policy_attachment" "ecs_exec_managed" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Task role: allow reading artifacts from S3 bucket (models, etc.)
resource "aws_iam_role" "ecs_task_role" {
  name               = "${local.name}-ecs-task-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume_role.json
  tags               = local.common_tags
}

resource "aws_iam_policy" "ecs_s3_read_policy" {
  name = "${local.name}-ecs-s3-read"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ListBucket"
        Effect   = "Allow"
        Action   = ["s3:ListBucket", "s3:GetBucketLocation"]
        Resource = [aws_s3_bucket.mlflow_artifacts.arn]
      },
      {
        Sid      = "ReadObjects"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = ["${aws_s3_bucket.mlflow_artifacts.arn}/*"]
      }
    ]
  })
  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "ecs_task_s3_read_attach" {
  role       = aws_iam_role.ecs_task_role.name
  policy_arn = aws_iam_policy.ecs_s3_read_policy.arn
}

# ALB + Target Group + Listener
resource "aws_lb" "app" {
  name               = "${local.name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = var.public_subnet_ids
  tags               = local.common_tags
}

resource "aws_lb_target_group" "api" {
  name        = "${local.name}-tg"
  port        = var.api_container_port
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = var.vpc_id

  health_check {
    path                = "/health"
    protocol            = "HTTP"
    matcher             = "200"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }

  tags = local.common_tags
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.app.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name}-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = tostring(var.api_cpu)
  memory                   = tostring(var.api_memory)
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn
  tags                     = local.common_tags

  container_definitions = jsonencode([
    {
      name      = "api"
      image     = "${aws_ecr_repository.app.repository_url}:${var.api_image_tag}"
      essential = true
      portMappings = [
        {
          containerPort = var.api_container_port
          hostPort      = var.api_container_port
          protocol      = "tcp"
        }
      ]
      environment = [
        {
          name  = "MLFLOW_TRACKING_URI"
          value = "http://${aws_instance.mlflow.private_ip}:${var.mlflow_port}"
        },
        { name = "MODEL_NAME", value = var.model_name },
        { name = "MODEL_ALIAS", value = var.model_alias }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.api.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}

# ECS Service
resource "aws_ecs_service" "app" {
  name            = "${local.name}-service"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.api_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.public_subnet_ids
    security_groups  = [aws_security_group.ecs_tasks_sg.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = var.api_container_port
  }

  depends_on = [aws_lb_listener.http]
  tags       = local.common_tags
}
