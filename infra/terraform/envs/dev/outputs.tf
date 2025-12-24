output "mlflow_public_ip" {
  value = aws_instance.mlflow.public_ip
}

output "mlflow_tracking_uri" {
  value = "http://${aws_instance.mlflow.public_ip}:5000"
}

output "mlflow_s3_bucket" {
  value = aws_s3_bucket.mlflow_artifacts.bucket
}

output "ecr_repo_url" {
  value = aws_ecr_repository.app.repository_url
}

output "alb_dns_name" {
  value = aws_lb.app.dns_name
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.this.name
}

output "ecs_service_name" {
  value = aws_ecs_service.app.name
}
