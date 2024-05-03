#!/bin/bash

# Download models
# wget -P consistency_models/models https://openaipublic.blob.core.windows.net/consistency/edm_imagenet64_ema.pt
# wget -P consistency_models/models https://openaipublic.blob.core.windows.net/consistency/cd_imagenet64_l2.pt
# wget -P consistency_models/models https://openaipublic.blob.core.windows.net/consistency/cd_imagenet64_lpips.pt
# wget -P consistency_models/models https://openaipublic.blob.core.windows.net/consistency/ct_imagenet64.pt

# wget -P consistency_models/models https://openaipublic.blob.core.windows.net/consistency/edm_bedroom256_ema.pt
wget -P consistency_models/models https://openaipublic.blob.core.windows.net/consistency/cd_bedroom256_l2.pt
wget -P consistency_models/models https://openaipublic.blob.core.windows.net/consistency/cd_bedroom256_lpips.pt
wget -P consistency_models/models https://openaipublic.blob.core.windows.net/consistency/ct_bedroom256.pt

wget -P consistency_models/models https://openaipublic.blob.core.windows.net/consistency/edm_cat256_ema.pt
wget -P consistency_models/models https://openaipublic.blob.core.windows.net/consistency/cd_cat256_l2.pt
wget -P consistency_models/models https://openaipublic.blob.core.windows.net/consistency/cd_cat256_lpips.pt
wget -P consistency_models/models https://openaipublic.blob.core.windows.net/consistency/ct_cat256.pt