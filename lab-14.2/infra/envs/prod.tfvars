# Entorno de producción: solo recibe lo que pasó por staging y fue aprobado
environment = "prod"

vpc_cidr            = "10.20.0.0/16"
public_subnet_cidrs = ["10.20.1.0/24", "10.20.2.0/24"]

task_cpu      = 512
task_memory   = 1024
desired_count = 1
