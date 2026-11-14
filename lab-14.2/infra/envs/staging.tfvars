# Entorno de staging: recibe cada commit de develop y de main
environment = "staging"

vpc_cidr            = "10.10.0.0/16"
public_subnet_cidrs = ["10.10.1.0/24", "10.10.2.0/24"]

task_cpu      = 256
task_memory   = 512
desired_count = 1
