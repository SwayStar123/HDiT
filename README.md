An attempt at a clean implemenation of HDiT (Hourglass Diffusion Transformers) krowson et al

This repo is based on REPA/SRA, but only implements Flow matching to keep it simple

install natten  with
`pip install natten==0.21.0+torch270cu128 -f https://whl.natten.org`



XL/2 with latent space with BS 32 (single gpu)

Inception Score: 19.498905181884766
FID: 64.62433915263483
sFID: 9.93260851526577
Precision: 0.42144
Recall: 0.4954

DiTXL baseline 19.5
SiTXL baseline: 17.2

considering the 8x smaller batchsize i think we can say this is roughly equivalent to baseline whilst being cheaper