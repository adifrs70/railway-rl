# Aplicații ale IA în sistemele de comandă-control ale traficului feroviar

Lucrare de disertație — BIOSINF, UPB-ETTI, 2026

## Structură proiect

project_diseratie/
├── graph/
│   ├── __init__.py
│   └── theia.py
├── simulation/
│   ├── __init__.py
│   ├── scenario_generator.py
│   ├── rail_env_theia.py
│   ├── train_dqn_theia.py
│   ├── train_ppo_theia.py
│   ├── evaluate_fixed_seeds.py
│   └── demo_theia.py
└── README.md

## Cerințe

pip install stable-baselines3 gymnasium networkx matplotlib numpy

## Rulare

cd simulation
python train_dqn_theia.py --level 2 --timesteps 100000
python evaluate_fixed_seeds.py --model dqn --n 1000
python demo_theia.py