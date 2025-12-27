variable "project_name" {
  description = "Project name prefix for resources"
  type        = string
  default     = "stock-price-predictor"
}

variable "aws_region" {
  description = "AWS region, e.g. us-west-2"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "public_subnet_ids" {
  description = "Two+ public subnet IDs (for ALB/ECS)"
  type        = list(string)
}

variable "public_subnet_id" {
  description = "A single public subnet ID (for MLflow EC2)"
  type        = string
}

variable "allowed_ssh_cidr" {
  description = "CIDR allowed to SSH to EC2 (your IP/32)"
  type        = string
}

variable "allowed_mlflow_cidr" {
  description = "CIDR allowed to access MLflow UI/API (your IP/32)"
  type        = string
}

variable "mlflow_key_name" {
  description = "EC2 key pair name for MLflow instance"
  type        = string
}

variable "mlflow_port" {
  description = "MLflow server port"
  type        = number
  default     = 5000
}

variable "api_container_port" {
  description = "FastAPI container port"
  type        = number
  default     = 8000
}

variable "api_desired_count" {
  description = "Desired number of ECS tasks"
  type        = number
  default     = 1
}

variable "api_cpu" {
  description = "Fargate CPU units"
  type        = number
  default     = 512
}

variable "api_memory" {
  description = "Fargate memory (MiB)"
  type        = number
  default     = 1024
}

variable "api_image_tag" {
  description = "ECR image tag used by ECS (you can keep latest, or set to git SHA)"
  type        = string
  default     = "latest"
}

variable "model_name" {
  description = "MLflow registered model name"
  type        = string
  default     = "stock-price-predictor"
}

variable "model_alias" {
  description = "MLflow alias to serve (e.g., prod)"
  type        = string
  default     = "prod"
}
