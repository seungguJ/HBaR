#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# Combining HBaR with adversarial training 

dataset=svhn
model=wideresnet-28-10

xw=1
lx=0.0001
ly=0.0005

run_hbar -cfg config/general-hbar-xentropy-cifar10.yaml -slmo -xw $xw -lx ${lx} -ly ${ly} -adv -mf ${dataset}_${model}_xw_${xw}_lx_${lx}_ly_${ly}.pt