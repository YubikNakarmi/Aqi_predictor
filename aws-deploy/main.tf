terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# ---------- Latest Ubuntu 22.04 AMI -----------------------------------------
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# ---------- Key Pair ---------------------------------------------------------
resource "aws_key_pair" "aqi_key" {
  key_name   = "aqi-key"
  public_key = file(var.public_key_path)
}

# ---------- Security Group ---------------------------------------------------
resource "aws_security_group" "aqi_sg" {
  name        = "aqi-security-group"
  description = "AQI Predictor - SSH, Airflow, MLflow, Optuna, Model Serve"

  # SSH
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }

  # Airflow Web UI
  ingress {
    description = "Airflow"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }

  # MLflow
  ingress {
    description = "MLflow"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }

  # Model Serve
  ingress {
    description = "Model Serve"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }

  # Optuna Dashboard
  ingress {
    description = "Optuna Dashboard"
    from_port   = 8081
    to_port     = 8081
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }

  # Flower (Celery monitor)
  ingress {
    description = "Flower"
    from_port   = 5555
    to_port     = 5555
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "aqi-sg" }
}

# ---------- EC2 Instance -----------------------------------------------------
resource "aws_instance" "aqi_ec2" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.aqi_key.key_name
  vpc_security_group_ids = [aws_security_group.aqi_sg.id]

  root_block_device {
    volume_size = var.volume_size
    volume_type = "gp3"
    encrypted   = true
  }

  user_data = <<-EOF
    #!/bin/bash
    apt update -y && apt upgrade -y
    apt install -y docker.io git
    systemctl enable docker && systemctl start docker
    usermod -aG docker ubuntu

    # Docker Compose standalone
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
  EOF

  # Copy GitHub deploy key to EC2
  provisioner "file" {
    source      = var.github_deploy_key_path
    destination = "/home/ubuntu/.ssh/github_key"

    connection {
      type        = "ssh"
      user        = "ubuntu"
      private_key = file(var.ec2_private_key_path)
      host        = self.public_ip
    }
  }

  # Setup SSH and clone repo
  provisioner "remote-exec" {
    inline = [
      "chmod 600 /home/ubuntu/.ssh/github_key",
      "echo -e 'Host github.com\\n  StrictHostKeyChecking no\\n  IdentityFile /home/ubuntu/.ssh/github_key' >> /home/ubuntu/.ssh/config",
      "chmod 600 /home/ubuntu/.ssh/config",
      "cd /home/ubuntu && git clone git@github.com:YubikNakarmi/Aqi_predictor.git",
    ]

    connection {
      type        = "ssh"
      user        = "ubuntu"
      private_key = file(var.ec2_private_key_path)
      host        = self.public_ip
    }
  }

  tags = {
    Name    = "AQI-Predictor"
    Project = "aqi-predictor"
  }
}

# ---------- Outputs ----------------------------------------------------------
output "public_ip" {
  value = aws_instance.aqi_ec2.public_ip
}

output "ssh_command" {
  value = "ssh -i <your-key>.pem ubuntu@${aws_instance.aqi_ec2.public_ip}"
}

output "airflow_url" {
  value = "http://${aws_instance.aqi_ec2.public_ip}:8080"
}

output "mlflow_url" {
  value = "http://${aws_instance.aqi_ec2.public_ip}:5000"
}

output "optuna_url" {
  value = "http://${aws_instance.aqi_ec2.public_ip}:8081"
}