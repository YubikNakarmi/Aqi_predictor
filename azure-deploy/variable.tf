variable "resource_group_name" {
  description = "Resource grp name"
  type        = string
}

variable "location" {
  description = "Location for the resource group"
  type        = string
  default     = "eastus"
}

variable "vmname" {
  description = "Name of the virtual machine"
  type        = string
  default     = "aqi-vm"
}

variable "admin-name" {
    description = "User name for VM admin"
    type        = string
    default     = "azureuser"
  
}

variable "my-ip" {
    description = "Your public IP address for NSG rule (CIDR format)"
    type        = string
    default     = "0.0.0.0/0"
}