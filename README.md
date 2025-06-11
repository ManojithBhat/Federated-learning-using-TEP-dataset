# Federated Learning System

This repository contains a federated learning setup with one server and two clients, each operating in its own isolated Python virtual environment. The system is designed for local testing and experimentation with federated learning workflows.

---

## Directory Structure

```
├── server/
│   ├── server.py
│   └── venv/
├── client1/
│   ├── client.py
│   ├── model_lstm.py
│   └── venv/
├── client2/
│   ├── client.py
│   ├── model_lstm.py
│   └── venv/
├── TEP_data.csv
├── prepare_data.py
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone the Repository

### 2. Set Up Virtual Environments and Install Dependencies

For each folder (`server`, `client1`, `client2`), follow these steps:

```bash
cd folder_name  # Replace with server, client1, or client2
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r ../requirements.txt
cd ..
```

### 3. Prepare the Dataset

You need to prepare the TEP dataset for your federated learning system. There are two ways to do this:

#### Option 1: Using existing TEP data files

If you have TEP data files (RData or CSV format), you can use them to prepare the dataset:

```bash
python prepare_data.py --input_dir path/to/your/TEP/data
```

Then copy the generated TEP_data.csv to both client directories:

```bash
copy TEP_data.csv client1\
copy TEP_data.csv client2\
```

#### Option 2: Generate synthetic data for testing

If you don't have the TEP data files, generate synthetic data and copy to each client folder:

```bash
python prepare_data.py
copy TEP_data.csv client1\
copy TEP_data.csv client2\
```

This will create a `TEP_data.csv` file that needs to be copied to each client directory.

---

## Running the System

### Step 1: Start the Server

```bash
cd server
source venv/bin/activate  # On Windows: venv\Scripts\activate
python server.py
```

### Step 2: Start Each Client

In separate terminals, run the following for each client:

**Client 1:**

```bash
cd client1
source venv/bin/activate  # On Windows: venv\Scripts\activate
python client.py --server_address 127.0.0.1:8080
```

**Client 2:**

```bash
cd client2
source venv/bin/activate  # On Windows: venv\Scripts\activate
python client.py --server_address 127.0.0.1:8080
```

---

## Model and Data Information

The system uses a Long Short-Term Memory (LSTM) neural network model for fault detection in the Tennessee Eastman Process (TEP) dataset. The model architecture consists of:

1. Two LSTM layers with 64 hidden units
2. A fully connected output layer for binary classification (fault/no-fault)

Each client:
1. Loads a different partition of the TEP dataset
2. Preprocesses the data using standardization and sliding windows
3. Trains the model locally during each federated learning round
4. Evaluates the model on a validation set

The server:
1. Coordinates the federated learning process
2. Aggregates model parameters using weighted averaging based on dataset sizes
3. Distributes the updated global model to clients for the next round

---

## Notes

* Ensure that the server is running before starting the clients
* Modify `--server_address` if the server is hosted on a different IP or port
* Do **not** commit `venv/` folders. They are excluded via `.gitignore`


