#!/usr/bin/env bash
# Provision the VM that hosts the MLflow tracking server (Lab 2). COSTS MONEY while running.
#
#   ./provision_azure.sh
#
# Creates, all tagged course=itcs355 student=<id> lab=2:
#   - Ubuntu 24.04 VM, Standard_B1s, public DNS name <DNS_LABEL>.<REGION>.cloudapp.azure.com
#   - inbound 80/443 open (HTTPS + certificate issuance); SSH only from this machine's IP
#   - system-assigned identity with Storage Blob Data Contributor on the course storage account,
#     so the server can write artifacts without a storage key
#   - a daily auto-shutdown, so a forgotten server stops billing compute overnight
# The VM installs Docker on first boot (cloud-init below); this folder is copied to it with scp.
# Deallocate it whenever you are not using it:  az vm deallocate -g <rg> -n <vm>
set -euo pipefail

RG="${RG:-itcs355-6688067}"
REGION="${REGION:-eastasia}"
VM="${VM:-itcs355-mlflow}"
DNS_LABEL="${DNS_LABEL:-itcs3556688067-mlflow}"
STORAGE="${STORAGE:-itcs3556688067}"
SHUTDOWN_UTC="${SHUTDOWN_UTC:-1600}"      # 16:00 UTC = 23:00 in Bangkok
TAGS=(course=itcs355 student=6688067 lab=2)

MY_IP="$(curl -s https://api.ipify.org)"
CLOUD_INIT="$(mktemp)"
cat > "$CLOUD_INIT" <<YAML
#cloud-config
package_update: true
packages: [docker.io, docker-compose-v2]
runcmd:
  - fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  - echo '/swapfile none swap sw 0 0' >> /etc/fstab
  - usermod -aG docker azureuser
YAML

echo "== VM =="
az vm create -g "$RG" -n "$VM" -l "$REGION" \
  --image Ubuntu2404 --size Standard_B1s \
  --admin-username azureuser --generate-ssh-keys \
  --public-ip-sku Standard --public-ip-address-dns-name "$DNS_LABEL" \
  --assign-identity '[system]' \
  --nsg-rule NONE \
  --custom-data "$CLOUD_INIT" \
  --tags "${TAGS[@]}" -o table

NSG="${VM}NSG"
echo "== network: 80/443 open, 22 from ${MY_IP} only =="
az network nsg rule create -g "$RG" --nsg-name "$NSG" -n allow-https --priority 1000 \
  --access Allow --protocol Tcp --direction Inbound --destination-port-ranges 80 443 -o none
az network nsg rule create -g "$RG" --nsg-name "$NSG" -n allow-ssh-me --priority 1010 \
  --access Allow --protocol Tcp --direction Inbound --destination-port-ranges 22 \
  --source-address-prefixes "${MY_IP}/32" -o none

echo "== identity: write artifacts to Blob Storage =="
PRINCIPAL="$(az vm show -g "$RG" -n "$VM" --query identity.principalId -o tsv)"
SCOPE="$(az storage account show -g "$RG" -n "$STORAGE" --query id -o tsv)"
az role assignment create --assignee-object-id "$PRINCIPAL" --assignee-principal-type ServicePrincipal \
  --role "Storage Blob Data Contributor" --scope "$SCOPE" -o none

echo "== daily auto-shutdown at ${SHUTDOWN_UTC} UTC =="
az vm auto-shutdown -g "$RG" -n "$VM" --time "$SHUTDOWN_UTC" -o none

# Tag everything az vm create made alongside the VM (disk, NIC, public IP, NSG, VNet).
for id in $(az resource list -g "$RG" --query "[?contains(name, '${VM}')].id" -o tsv); do
  az tag update --resource-id "$id" --operation merge --tags "${TAGS[@]}" -o none
done

FQDN="${DNS_LABEL}.${REGION}.cloudapp.azure.com"
rm -f "$CLOUD_INIT"
cat <<NEXT

Provisioned. Server hostname: ${FQDN}
Next, from the repository root on this machine:
  scp -r infra/tracking-server azureuser@${FQDN}:
  ssh azureuser@${FQDN}
  cd tracking-server
  ./make_secrets.sh ${FQDN} wasbs://itcs355@${STORAGE}.blob.core.windows.net/itcs355/mlruns
  docker compose up -d --build
Then set in your local cloud.env:
  MLFLOW_TRACKING_URI=https://${FQDN}
  MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD  (printed by make_secrets.sh)
NEXT
