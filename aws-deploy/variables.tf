variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "public_key_path" {
  description = "Path to your SSH public key file"
  type        = string
}

variable "my_ip" {
  description = "Your IP in CIDR format (e.g. 203.0.113.50/32). Run: curl ifconfig.me"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.medium"
}

variable "volume_size" {
  description = "Root EBS volume size in GB"
  type        = number
  default     = 30
}

variable "github_repo" {
  description = "GitHub repo SSH URL to clone"
  type        = string
  default     = "git@github.com:YubikNakarmi/Aqi_predictor.git"
}

variable "github_deploy_key_path" {
  description = "Path to the GitHub SSH deploy key (private key file)"
  type        = string
}

variable "ec2_private_key_path" {
  description = "Path to the EC2 SSH private key (.pem) for provisioner connections"
  type        = string
}

variable "github_branch" {
  description = "Git branch to checkout"
  type        = string
  default     = "test"
}