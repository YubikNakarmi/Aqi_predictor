

terraform{
    backend "azurerm" {
        resource_group_name  = "Pypipeline"
        storage_account_name = "pypipelineterraform"
        container_name       = "tfstate"
        key                  = "dev/terraform.tfstate"
        use_azuread_auth     = true
    }
}