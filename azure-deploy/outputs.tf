output "vm_public_ip" {
    value = azurerm_network_interface.nic.private_ip_address
}