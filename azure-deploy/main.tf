provider "azurerm" {
    
  features {}
  resource_provider_registrations = "none"

 
}

resource "azurerm_resource_group" "rg" {
    name     = var.resource_group_name
    location = var.location
}

# ------------------------- networks------------------------
resource "azurerm_virtual_network" "vnet" {
    name     = "${var.resource_group_name}-vnet"
    address_space = ["10.0.0.0/16"] #vnet address allocation
    location = azurerm_resource_group.rg.location
    resource_group_name = azurerm_resource_group.rg.name
    
    }



resource "azurerm_subnet" "subnet" {
    name = "${var.resource_group_name}-subnet"
    resource_group_name = azurerm_resource_group.rg.name
    virtual_network_name = azurerm_virtual_network.vnet.name
    address_prefixes = ["10.0.1.0/24"]#subnet address allocation
    }

resource "azurerm_public_ip" "pip" {
    name = "${var.resource_group_name}-pip"
    location = azurerm_resource_group.rg.location
    resource_group_name = azurerm_resource_group.rg.name
    allocation_method = "Static" #Standard SKU requires static allocation
  
}

resource "azurerm_network_interface" "nic" {
    name = "${var.resource_group_name}-nic"
    location = azurerm_resource_group.rg.location
    resource_group_name = azurerm_resource_group.rg.name

    ip_configuration {
        name = "internal"
        subnet_id = azurerm_subnet.subnet.id
        private_ip_address_allocation = "Dynamic"#dynamic ip to reduce cost
    }
  
}

#--------------------------- security grups ---------------
resource "azurerm_network_security_group" "nsg" {
    name = "${var.resource_group_name}-nsg"
    location = azurerm_resource_group.rg.location
    resource_group_name = azurerm_resource_group.rg.name
}

resource "azurerm_network_security_rule" "ssh" {
    name = "Allow-SSH"
    priority = 1001
    direction = "Inbound"
    access = "Allow"
    protocol = "Tcp"
    source_port_range = "*"
    destination_port_range = "22"
    source_address_prefix = "*"
    destination_address_prefix = "*"
    network_security_group_name = azurerm_network_security_group.nsg.name
    resource_group_name = azurerm_resource_group.rg.name
}

resource "azurerm_network_security_rule" "streamlit" {
    name = "Allow-Streamlit"
    priority = 1002
    direction = "Inbound"
    access = "Allow"
    protocol = "Tcp"
    source_port_range = "*"
    destination_port_range = "8501"
    source_address_prefix = "*"
    destination_address_prefix = "*"
    network_security_group_name = azurerm_network_security_group.nsg.name
    resource_group_name = azurerm_resource_group.rg.name
}

resource "azurerm_network_security_rule" "mlflow" {
    name = "Allow-MLflow"
    priority = 1003
    direction = "Inbound"
    access = "Allow"
    protocol = "Tcp"
    source_port_range = "*"
    destination_port_range = "5000"
    source_address_prefix = var.my-ip #restricting mlflow access to your public ip for security
    destination_address_prefix = "*"
    network_security_group_name = azurerm_network_security_group.nsg.name
    resource_group_name = azurerm_resource_group.rg.name
} 

resource "azurerm_network_security_rule" "optuna" {
    name = "Allow-Optuna"
    priority = 1004
    direction = "Inbound"
    access = "Allow"
    protocol = "Tcp"
    source_port_range = "*"
    destination_port_range = "8080"
    source_address_prefix = var.my-ip #restricting optuna access to your public ip for security
    destination_address_prefix = "*"
    network_security_group_name = azurerm_network_security_group.nsg.name
    resource_group_name = azurerm_resource_group.rg.name
}

resource "azurerm_network_security_rule" "airflow"{
    name = "Allow-Airflow"
    priority = 1005
    direction = "Inbound"
    access = "Allow"
    protocol = "Tcp"
    source_port_range = "*"
    destination_port_range = "8080"
    source_address_prefix = var.my-ip #restricting airflow access to your public ip for security
    destination_address_prefix = "*"
    network_security_group_name = azurerm_network_security_group.nsg.name
    resource_group_name = azurerm_resource_group.rg.name
}


#--------------------------vm------------------------------
resource "azurerm_linux_virtual_machine" "vm" {
    name = "${var.resource_group_name}-vm"
    resource_group_name = azurerm_resource_group.rg.name
    location = azurerm_resource_group.rg.location
    size = "Standard_B2s" #smallest vm size to reduce cost (2vcps, 4gib)
    admin_username = var.admin-name
    network_interface_ids = [azurerm_network_interface.nic.id]

    admin_ssh_key {
      username = var.admin-name
      public_key =  file("~/.ssh/id_ed25519.pub") #using existing ssh key for authentication
    }
    os_disk {
        caching = "ReadWrite"
        storage_account_type = "Standard_LRS"
    }
    source_image_reference {
        publisher = "Canonical"
        offer = "ubuntu-24_04-lts"
        sku = "server"
        version = "latest"
    }

    custom_data = filebase64("${path.module}/setup.sh") #cloud init script to install docker and start the container

    lifecycle {
      ignore_changes = [custom_data]
    }
}