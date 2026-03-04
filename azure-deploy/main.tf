provider "azurerm" {

    features {}# Automatically authenticates, using azure cli (az login) for credentials
  
}

resource "azurerm_resource_group" "rg" {
    name     = var.resource_group_name
    location = var.location
}


resource "azurerm_virtualnetwork" "vnet" {
    name     = "${var.resource_group_name}-vnet"
    address_space = ["10.0.0.0/16"] #vnet address allocation
    location = azurerm_resource_group.rg.location
    resource_group_name = azurerm_resource_group.rg.name
    }
    