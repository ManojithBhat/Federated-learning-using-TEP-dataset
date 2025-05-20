# Federated Learning System

This repository contains a federated learning setup with one server and two clients, each operating in its own isolated Python virtual environment. The system is designed for local testing and experimentation with federated learning workflows.

---

## Directory Structure

```
├── server/
│   ├── server.py
│   ├── requirements.txt
│   └── venv/
├── client1/
│   ├── client1.py
│   ├── requirements.txt
    └──model_lstm.py
│   └── venv/
├── client2/
│   ├── client2.py
│   ├── requirements.txt
│   └──model_lstm.py
│   └── venv/
```

---

## Setup Instructions

### 1. Clone the Repository
I hope you guys know how to do it 

### 2. Set Up Virtual Environments and Install Dependencies

For each folder (`server`, `client1`, `client2`), follow these steps:

```bash
cd folder_name  # Replace with server, client1, or client2
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

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
source venv/bin/activate
python client1.py --server_address 127.0.0.1:8080
```

**Client 2:**

```bash
cd client2
source venv/bin/activate
python client2.py --server_address 127.0.0.1:8080
```

---

## Notes

* Ensure that the server is running before starting the clients.
* Modify `--server_address` if the server is hosted on a different IP or port.
* Do **not** commit `venv/` folders. They are excluded via `.gitignore`.


