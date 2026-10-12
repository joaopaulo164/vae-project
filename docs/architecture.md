# Arquitetura do Modelo VAE

## Variational Autoencoder (VAE)

O modelo principal é um VAE totalmente conectado implementado em
\src/vae_project/models/vae.py\.

## Componentes Obrigatórios

| Componente | Variável | Módulo |
|-----------|---------|--------|
| Encoder MLP | \encoder\ | \models/components.py\ |
| Camada de média latente | \latent_mean_layer\ | \models/vae.py\ |
| Camada de log-variância | \latent_log_variance_layer\ | \models/vae.py\ |
| Trick de reparametrização | \eparameterize()\ | \models/vae.py\ |
| Amostra latente | \latent_sample\ | \models/vae.py\ |
| Decoder MLP | \decoder\ | \models/components.py\ |
| Reconstruction loss | \compute_reconstruction_loss()\ | \	raining/losses.py\ |
| KL divergence loss | \compute_kl_divergence_loss()\ | \	raining/losses.py\ |
| Total VAE loss | \compute_vae_loss()\ | \	raining/losses.py\ |

## Topologias Configuráveis

| Nome | hidden_dimensions | Parâmetros aprox. (latent=16) |
|------|-----------------|-------------------------------|
| small | [256, 128] | ~400K |
| medium | [512, 256] | ~1.2M |
| deep | [512, 256, 128] | ~1.5M |

O decoder é construído de forma simétrica (hidden_dimensions invertido).

## Saída do Forward Pass

O método \orward()\ retorna um \VAEOutput\ com:
- \econstruction\: imagem reconstruída em [0, 1]
- \latent_mean\: µ da posterior q(z|x)
- \latent_log_variance\: log s² da posterior
- \latent_sample\: amostra z via reparametrização

## Cabeça Supervisionada (opcional)

Implementada em \models/supervised_head.py\. Ativada por
\supervised_component.enabled: true\. Recebe \latent_mean\ como entrada
e produz logits de classificação.
