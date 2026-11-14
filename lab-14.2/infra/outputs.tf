output "environment" {
  description = "Entorno desplegado"
  value       = var.environment
}

output "image" {
  description = "Imagen con la que Terraform creó el servicio"
  value       = local.image
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "ECS service name"
  value       = aws_ecs_service.app.name
}

output "api_url" {
  description = "Public API endpoint via ALB"
  value       = "http://${aws_lb.main.dns_name}"
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group for container logs"
  value       = aws_cloudwatch_log_group.app.name
}
