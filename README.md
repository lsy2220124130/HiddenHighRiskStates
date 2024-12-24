# Hidden high-risk states identification from routine urban traffic

## Project Overview
This project identifies hidden high-risk states from routine urban traffic using a Maximum Entropy Model. Based on the maximum entropy model, we infer the underlying interaction network from complicated dynamical processes of urban traffic and construct the system energy landscape. In this way, we can locate hidden high-risk states that have never been observed from real data. The code generates four key figures (Figure 1 to Figure 4) described in the paper:

**1. Pairwise maximum entropy model for urban traffic (Fig. 1)**: Inferring the underlying interaction network of urban traffic and validating the model performance.

**2. State space analysis based on pairwise maximum entropy model (Fig. 2)**: Analyzing the energy distribution of system states.

**3. System energy landscape (Fig. 3)**: Vulnerability origin from system energy landscape construction.

**4. Hidden high-risk states identification (Fig. 4)**: Identifying hidden high-risk states based on a proposed risk indicator.

## Installation Instructions

### 1. Clone the Repository

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/lsy2220124130/HiddenHighRiskStates.git
cd HiddenHighRiskStates
```

### 2. Install Dependencies
Ensure you have Python 3.6+ installed. You can create a virtual environment and install the necessary dependencies:

```bash
pip install -r requirements.txt
```

### 3. Prepare the Input Data
You will need the following raw data files to run the code:
- raw_data/450-510-global_state_day_time_0.25_0.09.csv
- raw_data/hex_topology_inter_box.xlsx
- raw_data/dict_cell_geo.json

These files contain the data necessary for the model and energy landscape analysis.

## Usage
To run the code, simply execute the individual scripts corresponding to each part of the process. The following commands will generate the required figures:

**1. Step 1**: Obtain model parameters:

```bash
python 0-learn_hi_Jij.py
```

**2. Step 2**: Generate intermediate results (this step can take some time):

```bash
python 0-each_global_state_lcc_jam_lcc.py
python 1-each_global_state_e.py
```

**3. Step 3**: Generate Figure 1 (Maximum Entropy Model and Model Validation):

```bash
python fig_1.py
```

**4. Step 4**: Generate Figure 2 (Energy Distribution):

```bash
python fig_2.py
```

**5. Step 5**: Generate Figure 3 (System Energy Landscape):

```bash
python fig_3.py
```

**6. Step 6**: Generate Figure 4 (Hidden High-Risk States Identification):

```bash
python fig_4.py
```

After running the scripts, the output figures will be displayed.
