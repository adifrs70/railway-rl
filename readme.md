# Aplicații ale Inteligenței Artificiale în Sistemele de Comandă-Control ale Traficului Feroviar

Lucrare de disertație — BIOSINF, UPB-ETTI, 2026

Sistem experimental de suport decizional pentru alegerea parcursurilor într-o stație feroviară simulată, utilizând metode de învățare prin întărire profundă (DRL). Stația analizată — **Theia** — este reprezentată printr-un model grafic orientat, iar deciziile agentului sunt validate determinist prin structura grafului.

## Prezentare generală

Proiectul implementează și evaluează comparativ doi agenți de învățare prin întărire profundă pentru gestionarea conflictelor de parcurs feroviar:

- **Model grafic al stației Theia**: reprezentare prin graf orientat (NetworkX DiGraph), cu 42 de noduri — capete de acces, macazuri, segmente de cale, semnale terminale
- **Generator de scenarii**: producere procedurală de situații operaționale cu niveluri controlabile de dificultate, defecte de infrastructură și parcursuri active
- **Mediu RL (TheiaCFEnv)**: interfață Gymnasium-compatibilă, spațiu de observații `Box(42,)`, spațiu de acțiuni `Discrete(6)`
- **Agent DQN**: algoritm off-policy bazat pe estimarea valorii acțiunilor — rată de succes **88%** pe 1000 de scenarii fixe
- **Agent PPO**: algoritm on-policy bazat pe optimizarea directă a politicii — rată de succes **77%** pe același set de scenarii

Validarea este realizată din două perspective complementare: cantitativă (pe seed-uri fixe rezervate exclusiv evaluării) și calitativă (scenarii demonstrative cu vizualizare grafică matplotlib).

## Structura proiectului

```
project_diseratie/
├── graph/
│   ├── __init__.py
│   └── theia.py                   # Graf orientat stație, logică operațională
├── simulation/
│   ├── __init__.py
│   ├── scenario_generator.py      # Generator scenarii cu sampling balansat
│   ├── rail_env_theia.py          # Mediu RL Gymnasium-compatibil
│   ├── train_dqn_theia.py         # Antrenament DQN (Stable-Baselines3)
│   ├── train_ppo_theia.py         # Antrenament PPO (Stable-Baselines3)
│   ├── evaluate_fixed_seeds.py    # Validare cantitativă pe seed-uri fixe
│   └── demo_theia.py              # Demo interactiv cu vizualizare grafică
└── README.md
```

## Cerințe

- Python 3.10 sau mai nou
- stable-baselines3 >= 2.8.0
- gymnasium
- networkx
- matplotlib
- numpy

## Instalare

1. Clonează repository-ul:
```bash
git clone https://github.com/USERNAME/railway-rl-theia.git
cd railway-rl-theia
```

2. Creează și activează un mediu virtual (recomandat):
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```
### 1. Antrenare DQN

```bash
# Nivel 1 — scenarii simple
python train_dqn_theia.py --level 1 --timesteps 100000

# Nivel 2 — scenarii cu defecte și trafic activ
python train_dqn_theia.py --level 2 --timesteps 100000
```

Rezultate salvate în:
- `models/dqn_level1_best/best_model.zip`
- `models/dqn_level2_best/best_model.zip`
- `logs/dqn_level2_convergenta.png` — grafic convergență
- `logs/dqn_level2_eval_summary.json` — evaluare finală

### 2. Antrenare PPO

```bash
# Nivel 1
python train_ppo_theia.py --level 1 --timesteps 150000

# Nivel 2
python train_ppo_theia.py --level 2 --timesteps 300000
```

### 3. Validare cantitativă pe seed-uri fixe

```bash
# Evaluare DQN
python evaluate_fixed_seeds.py --model dqn --n 1000 --difficulty 2

# Evaluare PPO
python evaluate_fixed_seeds.py --model ppo --n 1000 --difficulty 2

# Comparație DQN vs PPO
python evaluate_fixed_seeds.py --model both --n 1000 --difficulty 2
```

Intervalul de seed-uri utilizat (100000–100999) este rezervat exclusiv evaluării finale și separat de procesul de antrenare.

Rezultate salvate în `logs/`:
- `eval_fixed_seeds_dqn.json`
- `eval_fixed_seeds_ppo.json`
- `eval_fixed_seeds_comparison.json`

### 4. Demo interactiv — validare calitativă

```bash
# Cu model DQN (implicit)
python demo_theia.py

# Cu model PPO
python demo_theia.py --model ppo

# Fără model — logică directă din graf (toate tipurile de cerere disponibile)
python demo_theia.py --no-model
```

**Flux demo:**
1. Utilizatorul introduce defectele de infrastructură (ex: `M8 defect`, `M3 blocat minus`)
2. Utilizatorul introduce cererea trenului (ex: `Y -> LII`)
3. Sistemul decide aleatoriu dacă există un parcurs activ în stație
4. Modelul analizează starea și decide
5. Vizualizare grafică matplotlib

**Tipuri de cerere disponibile cu model RL:**
```
Intrare cu oprire:  X->L1, X->LII, X->L3, X->L4
                    Y->L1, Y->LII, Y->L3, Y->L4
Tranzit complet:    X->Y, Y->X
```

**Tipuri de cerere disponibile cu `--no-model`:**
```
+ Ieșire de pe linie: Y1->X, YII->X, Y3->X, Y4->X
                      X1->Y, XII->Y, X3->Y, X4->Y
```

## Algoritmi și arhitectură

### Spațiu de observații — `Box(42,)`

| Componente | Descriere |
|---|---|
| `[0..11]` | 6 macazuri × 2 atribute: poziție (plus/minus) și stare (operațional/blocat/defect) |
| `[12..37]` | 26 segmente × 1 atribut: stare (liber/rezervat/ocupat/indisponibil) |
| `[38]` | Originea cererii (index 0..9) |
| `[39]` | Destinația cererii (index 0..9) |
| `[40]` | Parcurs activ existent (0 sau 1) |
| `[41]` | Destinația parcursului activ (index 0..9) |

### Spațiu de acțiuni — `Discrete(6)`

| Cod | Acțiune | Semnificație |
|---|---|---|
| 0 | așteaptă | nu se acordă parcurs în starea curentă |
| 1 | L1 | parcurs asociat Liniei 1 |
| 2 | LII | parcurs asociat Liniei II (directă) |
| 3 | L3 | parcurs asociat Liniei 3 |
| 4 | L4 | parcurs asociat Liniei 4 |
| 5 | tranzit_capat | tranzit X→Y/Y→X sau ieșire semnal→capăt |

### Funcția de recompensă

| Etichetă decizie | Condiție | Recompensă |
|---|---|---|
| `parcurs_realizabil` | acțiunea corespunde cererii și ruta există | +10 |
| `asteapta_corect` | agentul așteaptă, nu există rută | +3 |
| `asteapta_gresit` | agentul așteaptă deși există rută | -10 |
| `actiune_nepotrivita` | acțiunea nu corespunde cererii | -10 |
| `ruta_imposibila` | acțiunea corectă dar ruta nu există în graf | -10 |

### Hiperparametri

| Parametru | DQN | PPO |
|---|---|---|
| Learning rate | 1e-3 | 3e-4 |
| Arhitectură rețea | MLP [64, 64] | MLP [64, 64] |
| Gamma | 0.99 | 0.99 |
| Replay buffer | 50.000 | — |
| n_steps | — | 512 |
| n_epochs | — | 10 |
| Clip range | — | 0.2 |
| Entropy coef | — | 0.01 |

## Rezultate

Evaluare finală pe **1000 de scenarii** de dificultate 2, seed-uri 100000–100999:

| Model | Reward mediu | Dev. standard | Rată decizii corecte | Rată eroare |
|---|---|---|---|---|
| **DQN** | **5.269** | 6.476 | **88.00%** | **12.00%** |
| PPO | 3.055 | 7.758 | 77.00% | 23.00% |

### Distribuția tipurilor de decizie

| Tip decizie | DQN | PPO |
|---|---|---|
| parcurs_realizabil (+10) | 547 (54.7%) | 435 (43.5%) |
| asteapta_corect (+3) | 333 (33.3%) | 335 (33.5%) |
| asteapta_gresit (-10) | 13 (1.3%) | 112 (11.2%) |
| actiune_nepotrivita (-10) | 47 (4.7%) | 70 (7.0%) |
| ruta_imposibila (-10) | 60 (6.0%) | 48 (4.8%) |

**DQN** obține performanța mai bună în această configurație. Spațiul discret redus de acțiuni și episoadele de un singur pas favorizează algoritmul off-policy cu replay buffer. **PPO** are un comportament mai conservator — alege mai frecvent așteptarea, ceea ce reduce rutele imposibile alese dar crește cazurile de așteptare greșită.

## Note metodologice

**Separarea antrenare / evaluare**: în antrenare, scenariile sunt generate aleatoriu fără seed fix. Intervalul de seed-uri 100000–100999 este rezervat exclusiv evaluării finale pentru a asigura reproductibilitatea și separarea de procesul de învățare.

**Rolul validatorului grafic**: agentul DQN/PPO nu construiește direct ruta fizică nod cu nod. El selectează o acțiune operațională (tipul de parcurs), iar ruta fizică concretă este determinată de validatorul grafic (`get_rute_valide`). Destinația cererii nu se schimbă niciodată — rerutarea înseamnă aceeași origine și destinație, pe o cale alternativă în graf.

**Validare calitativă vs. cantitativă**: `demo_theia.py` este destinat interpretării deciziilor agentului în scenarii concrete, nu calculului statistic al performanței. Dovada statistică provine din `evaluate_fixed_seeds.py`.

## Depanare

- **Eroare `observation space` incompatibil**: modelele vechi antrenate pe `OBS_DIM=40` sunt incompatibile cu mediul actual (`OBS_DIM=42`). Șterge modelele vechi și reantrenează.
- **Acțiunile 1-4 niciodată alese**: asigură-te că folosești `scenario_generator.py` cu sampling balansat pe tipuri de acțiuni. Fără el, acțiunile de tranzit domină distribuția și modelul ignoră liniile L1-L4.
- **Eroare `NoneType format`** în `evaluate_fixed_seeds.py`: actualizează la versiunea curentă care tratează explicit `ACTIUNE_LINIE[0] = None`.

## Contact

- **Mihai-Adrian Costică** — `adriancostica00@gmail.com`
- Coordonatori: Prof. dr. ing. Dragoș Burileanu, Șl. dr. ing. Șerban Mihalache
- Universitatea Națională de Știință și Tehnologie POLITEHNICA București
- Facultatea de Electronică, Telecomunicații și Tehnologia Informației
- Program masterat BIOSINF, 2026
