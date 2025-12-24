variable "aws_region" { type = string }

variable "project_name" { type = string default = "stock-price-predictor" }

variable "vpc_id" { type = string }
variable "public_subnet_id" { type = string }

variable "mlflow_instance_type" { type = string default = "t3.small" }
variable "mlflow_key_name" { type = string } # existing EC2 keypair name

variable "allowed_ssh_cidr" { type = string }     # e.g., "YOUR_PUBLIC_IP/32"
variable "allowed_mlflow_cidr" { type = string }  # e.g., "YOUR_PUBLIC_IP/32" or VPC CIDR

# --- Networking for ALB + ECS ---
variable "public_subnet_ids" {
  type        = list(string)
  description = "Public subnets for ALB (and ECS if you keep tasks public for MVP)."
}

# For MVP, you can run ECS tasks in public subnets with public IPs.
# For a more production setup, use private subnets + NAT and set assign_public_ip=false.
variable "ecs_subnet_ids" {
  type        = list(string)
  description = "Subnets where ECS tasks will run."
}

# --- ECS service config ---
variable "ecs_desired_count" {
  type    = number
  default = 1
}

variable "container_port" {
  type    = number
  default = 8000
}

variable "task_cpu" {
  type    = number
  default = 256
}

variable "task_memory" {
  type    = number
  default = 512
}

variable "image_tag" {
  type        = string
  default     = "latest"
  description = "Default tag Terraform deploys. CI/CD will override this with git SHA."
}
