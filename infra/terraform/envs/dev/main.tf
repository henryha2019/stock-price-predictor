locals {
  name = var.project_name
}

# --- S3 bucket for MLflow artifacts ---
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket_prefix = "${local.name}-mlflow-artifacts-"
}

resource "aws_s3_bucket_versioning" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_public_access_block" "mlflow_artifacts" {
  bucket                  = aws_s3_bucket.mlflow_artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- IAM role for EC2 (MLflow server) ---
data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals { type = "Service" identifiers = ["ec2.amazonaws.com"] }
  }
}

resource "aws_iam_role" "mlflow_ec2_role" {
  name               = "${local.name}-mlflow-ec2-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json
}

data "aws_iam_policy_document" "mlflow_s3_policy" {
  statement {
    actions = ["s3:ListBucket"]
    resources = [aws_s3_bucket.mlflow_artifacts.arn]
  }
  statement {
    actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.mlflow_artifacts.arn}/*"]
  }
}

resource "aws_iam_policy" "mlflow_s3" {
  name   = "${local.name}-mlflow-s3"
  policy = data.aws_iam_policy_document.mlflow_s3_policy.json
}

resource "aws_iam_role_policy_attachment" "mlflow_ec2_s3_attach" {
  role       = aws_iam_role.mlflow_ec2_role.name
  policy_arn = aws_iam_policy.mlflow_s3.arn
}

resource "aws_iam_instance_profile" "mlflow_profile" {
  name = "${local.name}-mlflow-profile"
  role = aws_iam_role.mlflow_ec2_role.name
}

# --- Security group for MLflow EC2 ---
resource "aws_security_group" "mlflow_sg" {
  name        = "${local.name}-mlflow-sg"
  description = "MLflow server SG"
  vpc_id      = var.vpc_id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  ingress {
    description = "MLflow"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = [var.allowed_mlflow_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# --- AMI (Amazon Linux 2023) ---
data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}

# --- User data: install + run MLflow server ---
locals {
  user_data = <<-EOF
    #!/bin/bash
    set -e

    yum update -y
    yum install -y python3-pip
    pip3 install --upgrade pip
    pip3 install mlflow boto3

    mkdir -p /opt/mlflow
    cd /opt/mlflow

    cat > /etc/systemd/system/mlflow.service <<'SERVICE'
    [Unit]
    Description=MLflow Tracking Server
    After=network.target

    [Service]
    Type=simple
    Restart=always
    RestartSec=3
    WorkingDirectory=/opt/mlflow
    ExecStart=/usr/local/bin/mlflow server \
      --host 0.0.0.0 \
      --port 5000 \
      --backend-store-uri sqlite:////opt/mlflow/mlflow.db \
      --default-artifact-root s3://${aws_s3_bucket.mlflow_artifacts.bucket}

    [Install]
    WantedBy=multi-user.target
    SERVICE

    systemctl daemon-reload
    systemctl enable mlflow
    systemctl start mlflow
  EOF
}

resource "aws_instance" "mlflow" {
  ami                         = data.aws_ami.al2023.id
  instance_type               = var.mlflow_instance_type
  subnet_id                   = var.public_subnet_id
  vpc_security_group_ids      = [aws_security_group.mlflow_sg.id]
  iam_instance_profile        = aws_iam_instance_profile.mlflow_profile.name
  key_name                    = var.mlflow_key_name
  associate_public_ip_address = true

  user_data = local.user_data

  tags = { Name = "${local.name}-mlflow" }
}
