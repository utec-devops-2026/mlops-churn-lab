# Registro de imágenes compartido por staging y producción.
# Vive fuera de los entornos: el pipeline publica aquí la imagen de cada commit
# y los dos entornos leen del mismo repositorio. Se aplica una sola vez.

terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Nombre del repositorio; debe coincidir con project_name de los entornos"
  type        = string
  default     = "churn-predictor"
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project   = var.project_name
      ManagedBy = "terraform"
    }
  }
}

resource "aws_ecr_repository" "app" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Conserva las últimas 10 imágenes para controlar el costo de almacenamiento
resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

output "repository_url" {
  description = "URL del repositorio: el pipeline publica aquí y los entornos leen de aquí"
  value       = aws_ecr_repository.app.repository_url
}
