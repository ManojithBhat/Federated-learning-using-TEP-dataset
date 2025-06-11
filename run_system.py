import os
import time
import subprocess
import sys

def main():
    # Get the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Fix client1.py issue - client_id should be 1 not 2
    client1_path = os.path.join(base_dir, "client1", "client.py")
    client2_path = os.path.join(base_dir, "client2", "client.py")
    
    # Fix client IDs in client files if needed
    fix_client_ids(client1_path, client2_path)
    
    print("Starting server and client processes in separate windows...")
    
    # Start server in a new window
    server_dir = os.path.join(base_dir, "server")
    server_cmd = f'start "FL Server" cmd /k "cd /d {server_dir} && .\\venv\\Scripts\\activate && python server.py"'
    print("Launching server...")
    os.system(server_cmd)
    
    # Wait for server to initialize
    print("Waiting for server to initialize (5 seconds)...")
    time.sleep(5)
    
    # Start clients in separate windows
    client1_dir = os.path.join(base_dir, "client1")
    client1_cmd = f'start "FL Client 1" cmd /k "cd /d {client1_dir} && .\\venv\\Scripts\\activate && python client.py --server_address 127.0.0.1:8080"'
    print("Launching client 1...")
    os.system(client1_cmd)
    
    # Wait a moment between client launches
    time.sleep(1)
    
    client2_dir = os.path.join(base_dir, "client2")
    client2_cmd = f'start "FL Client 2" cmd /k "cd /d {client2_dir} && .\\venv\\Scripts\\activate && python client.py --server_address 127.0.0.1:8080"'
    print("Launching client 2...")
    os.system(client2_cmd)
    
    print("\nAll processes started in separate windows.")
    print("You can monitor the progress in each window.")
    print("Close the windows manually when the federated learning process completes.\n")

def fix_client_ids(client1_path, client2_path):
    """Fix client IDs if they're incorrect"""
    try:
        # Fix client1.py
        with open(client1_path, 'r') as file:
            content = file.read()
        
        if "client_id=2" in content:
            print("Fixing client ID in client1.py...")
            content = content.replace("client_id=2", "client_id=1")
            with open(client1_path, 'w') as file:
                file.write(content)
        
        # Fix client2.py
        with open(client2_path, 'r') as file:
            content = file.read()
        
        if "client_id=1" in content:
            print("Fixing client ID in client2.py...")
            content = content.replace("client_id=1", "client_id=2")
            with open(client2_path, 'w') as file:
                file.write(content)
        
    except Exception as e:
        print(f"Error fixing client IDs: {e}")
        # Continue anyway

if __name__ == "__main__":
    main()