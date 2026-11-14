#!/usr/bin/env bash
# CRISP-ML(Q) fase 5 · Despliega una imagen en el servicio ECS de un entorno.
#
# Uso: scripts/deploy_ecs.sh <staging|prod> <imagen>
#   scripts/deploy_ecs.sh staging 123456789012.dkr.ecr.us-east-1.amazonaws.com/churn-predictor:3f2a9c1
#
# Registra una revisión nueva de la task definition con esa imagen, actualiza el
# servicio y espera a que quede estable. El mismo comando sirve para el rollback:
# basta pasar la imagen del commit anterior.
set -euo pipefail

ENVIRONMENT="${1:?uso: deploy_ecs.sh <staging|prod> <imagen>}"
IMAGE="${2:?uso: deploy_ecs.sh <staging|prod> <imagen>}"
PROJECT="${PROJECT_NAME:-churn-predictor}"
REGION="${AWS_REGION:-us-east-1}"
CLUSTER="$PROJECT-$ENVIRONMENT"
SERVICE="$CLUSTER-service"

PREVIOUS=$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --region "$REGION" \
  --query 'services[0].taskDefinition' --output text) || PREVIOUS="None"
if [ "$PREVIOUS" = "None" ]; then
  echo "No existe el servicio $SERVICE en el cluster $CLUSTER: crea el entorno con Terraform (infra/)." >&2
  exit 1
fi

# La revisión nueva es una copia de la actual: solo cambia la imagen
NEW_DEFINITION=$(aws ecs describe-task-definition --task-definition "$PREVIOUS" --region "$REGION" \
  --query 'taskDefinition' \
  | jq --arg IMAGE "$IMAGE" \
      'del(.taskDefinitionArn, .revision, .status, .requiresAttributes, .compatibilities, .registeredAt, .registeredBy)
       | .containerDefinitions[0].image = $IMAGE')
NEW=$(aws ecs register-task-definition --cli-input-json "$NEW_DEFINITION" --region "$REGION" \
  --query 'taskDefinition.taskDefinitionArn' --output text)

echo "Entorno:  $ENVIRONMENT"
echo "Imagen:   $IMAGE"
echo "Anterior: $PREVIOUS"
echo "Nueva:    $NEW"

aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" --task-definition "$NEW" \
  --region "$REGION" --query 'service.serviceName' --output text
aws ecs wait services-stable --cluster "$CLUSTER" --services "$SERVICE" --region "$REGION"

# Si las tareas nuevas no pasaron el health check, el circuit breaker de ECS devolvió
# el servicio a la revisión anterior y "estable" no significa "desplegado"
RUNNING=$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --region "$REGION" \
  --query 'services[0].taskDefinition' --output text)
if [ "$RUNNING" != "$NEW" ]; then
  echo "El despliegue falló: ECS hizo rollback a $RUNNING" >&2
  exit 1
fi

URL="http://$(aws elbv2 describe-load-balancers --names "$CLUSTER-alb" --region "$REGION" \
  --query 'LoadBalancers[0].DNSName' --output text)"
echo "API:      $URL"

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  {
    echo "url=$URL"
    echo "task_definition=$NEW"
    echo "previous_task_definition=$PREVIOUS"
  } >> "$GITHUB_OUTPUT"
fi
