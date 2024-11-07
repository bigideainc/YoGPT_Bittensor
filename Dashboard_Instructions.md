# Instructions for Setting Up and Running the YoGPT Bittensor Dashboard

This document provides step-by-step instructions for setting up, running, and troubleshooting the YoGPT Bittensor Dashboard. Follow these steps to ensure a smooth experience.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Clone the Repository](#clone-the-repository)
3. [Set Up the Environment](#set-up-the-environment)
4. [Run the Validator](#run-the-validator)
5. [Run the Dashboard](#run-the-dashboard)
6. [Access the Dashboard](#access-the-dashboard)
7. [Troubleshooting](#troubleshooting)
8. [Updating and Committing Changes](#updating-and-committing-changes)

## Prerequisites

- Ensure you have the following installed:
  - Python 3.11 or higher
  - Git
  - Streamlit
  - WebSocket library

## Clone the Repository

1. Open your terminal.
2. Clone the repository using the following command:

   ```bash
   git clone https://github.com/bigideainc/YoGPT_Bittensor.git
   ```

3. Navigate to the project directory:

   ```bash
   cd YoGPT_Bittensor
   ```

## Set Up the Environment

1. **Create a Virtual Environment** (optional but recommended):

   ```bash
   python3 -m venv venv
   ```

2. **Activate the Virtual Environment**:

   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

3. **Install Required Packages**:

   Make sure to install the necessary packages listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

## Run the Validator

1. **First, run the validator to start and keep the WebSocket server running**:

   ```bash
   python3 neurons/validator.py --netuid 100 --subtensor.network test --wallet.name validator --wallet.hotkey default --logging.debug
   ```

   **Running as a module**:

   ```bash
   python3 -m neurons.validator --netuid 100 --subtensor.network test --wallet.name validator --wallet.hotkey default --logging.debug
   ```

2. **Secondly, run the Flask app**:

   ```bash
   python dashboard/app.py
   ```

3. **Third, run the Streamlit app**:

   ```bash
   streamlit run main.py
   ```

   Access it at: [http://localhost:5000](http://localhost:5000)

## Run the Dashboard

1. **Start the WebSocket Server** (if applicable):

   Ensure that the WebSocket server is running on `ws://localhost:8765`. This server should be set up to send metrics data to the dashboard.

2. **Run the Streamlit Dashboard**:

   Execute the following command to start the Streamlit application:

   ```bash
   streamlit run main.py --server.port 5000
   ```

## Access the Dashboard

1. Open your web browser.
2. Navigate to the following URL:

   ```
   http://localhost:5000
   ```

3. You should see the YoGPT Bittensor Dashboard interface.

## Troubleshooting

- **Connection Issues**: If you encounter connection errors with the WebSocket, ensure that the WebSocket server is running and accessible at the specified URI (`ws://localhost:8765`).
- **Module Not Found**: If you receive errors about missing modules, ensure all dependencies are installed correctly.
- **Error Logs**: Check the error logs for any issues related to file permissions or missing files.

## Updating and Committing Changes

1. **Make Your Changes**: Edit the files as needed in your project.
2. **Stage Your Changes**:

   ```bash
   git add .
   ```

3. **Commit Your Changes**:

   ```bash
   git commit -m "Your descriptive commit message"
   ```

4. **Push Changes to the Backport Branch** (if applicable):

   ```bash
   git push origin backport
   ```

5. **Merge Backport into Main**:

   Switch to the main branch:

   ```bash
   git checkout main
   ```

   Merge the backport branch:

   ```bash
   git merge backport
   ```

6. **Push Changes to the Main Branch**:

   ```bash
   git push -u origin main
   ```

## Configuration Files

### .replit

```plaintext
modules = ["python-3.11"]

[nix]
channel = "stable-24_05"

[workflows]
runButton = "Project"

[[workflows.workflow]]
name = "Project"
mode = "parallel"
author = "agent"

[[workflows.workflow.tasks]]
task = "workflow.run"
args = "Streamlit Dashboard"

[[workflows.workflow]]
name = "Streamlit Dashboard"
author = "agent"

[workflows.workflow.metadata]
agentRequireRestartOnSave = false

[[workflows.workflow.tasks]]
task = "packager.installForAll"

[[workflows.workflow.tasks]]
task = "shell.exec"
args = "streamlit run main.py --server.port 5000"
waitForPort = 5000

[deployment]
deploymentTarget = "gce"
run = ["sh", "-c", "python main.py"]

[[ports]]
localPort = 5000
externalPort = 80
```

### public/index.html

```html
<!DOCTYPE html>
<html data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YoGPT Bittensor Dashboard</title>
    <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
</head>
<body>
    <div class="container">
        <h1>YoGPT Bittensor Dashboard</h1>
        <p>This is a static entry point for the Streamlit application.</p>
        <p>Please access the dashboard at: <a href="/streamlit" class="btn btn-primary">Launch Dashboard</a></p>
    </div>
</body>
</html>
```

### templates/dashboard.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard</title>
    <script>
        async function fetchMetrics() {
            const response = await fetch('/metrics');
            const data = await response.json();
            document.getElementById('metrics').innerText = JSON.stringify(data, null, 2);
        }

        setInterval(fetchMetrics, 5000); // Fetch metrics every 5 seconds
    </script>
</head>
<body>
    <h1>Dashboard</h1>
    <pre id="metrics">Loading metrics...</pre>
</body>
</html>
```

### main.py

```python
import streamlit as st
import asyncio
import websockets
import json
from components.metrics import display_metrics
from components.map import display_map
from components.leaderboard import display_leaderboard
from components.alerts import check_network_alerts
from components.node_stats import display_node_stats
from components.historical_analysis import display_historical_analysis
from utils.data_generator import DataGenerator

st.set_page_config(
    page_title="YoGPT Bittensor Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_generator' not in st.session_state:
    st.session_state.data_generator = DataGenerator()

# Initialize metrics data
if 'metrics_data' not in st.session_state:
    st.session_state.metrics_data = {}

# Function to receive data from WebSocket
async def receive_metrics():
    uri = "ws://localhost:8765"  # WebSocket server URI
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    st.session_state.metrics_data.update(data)  # Update metrics data in session state
        except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError) as e:
            print(f"Connection error: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)  # Wait before trying to reconnect

# Run the WebSocket client in a separate thread
def run_websocket_client():
    asyncio.run(receive_metrics())

# Start the WebSocket client
if st.button("Start Listening"):
    run_websocket_client()

# Header
st.title("Alpha 9 Labs - Infrastructure for Decentralized Intelligence")
st.markdown("YoGPT Bittensor is revolutionizing decentralized AI training through the Bittensor network, enabling anyone to contribute compute and participate in the network's growth.")

# Main layout
col1, col2 = st.columns([3, 2])

with col1:
    # Training Progress
    progress = st.session_state.data_generator.get_progress()
    st.progress(progress['percentage'] / 100)
    st.write(f"{progress['percentage']}%")
    st.write(f"{progress['tokens']:,} / 1T tokens")
    
    # Metrics Section
    st.subheader("Real-time Metrics")
    st.write(st.session_state.metrics_data)  # Display the metrics data
    
    # Historical Analysis Section
    with st.expander("Historical Analysis & Predictions", expanded=True):
        display_historical_analysis()
    
    # Node Statistics Section
    with st.expander("Node Statistics", expanded=True):
        display_node_stats()

with col2:
    # Map
    display_map()
    
    # Alerts Section
    alert_system = check_network_alerts(st.session_state.data_generator)
    alert_system.display_alerts()
    
    # Leaderboard
    display_leaderboard()

# Call the historical analysis display function
display_historical_analysis()

# Auto-refresh every 5 seconds
if st.button("Refresh"):
    st.experimental_rerun()
```

### .local/state/replit/agent/repl_state.json

```json
{"repl_description":"A Streamlit-based dashboard for monitoring Bittensor network training progress with real-time metrics and visualizations","repl_description_state":"DESCRIPTION_APPROVED","repl_stack":null}
```

### pyproject.toml

```plaintext
[project]
name = "repl-nix-bittensordash"
version = "0.1.0"
description = "Add your description here"
requires-python = ">=3.11"
dependencies = [
    "numpy>=2.1.2",
    "pandas>=2.2.3",
    "plotly>=5.24.1",
    "scikit-learn>=1.5.2",
    "streamlit>=1.39.0",
]
```

### replit_zip_error_log.txt

```plaintext
{"error":".zip archives do not support non-regular files","level":"error","msg":"unable to write file .cache/replit/modules/python-3.11","time":"2024-11-02T09:17:17Z"}
{"error":".zip archives do not support non-regular files","level":"error","msg":"unable to write file .cache/replit/modules/replit","time":"2024-11-02T09:17:17Z"}
{"error":".zip archives do not support non-regular files","level":"error","msg":"unable to write file .cache/uv/wheels-v1/index/b2a7eb67d4c26b82/altair/altair-5.4.1-py3-none-any","time":"2024-11-02T09:17:41Z"}
{"error":".zip archives do not support non-regular files","level":"error","msg":"unable to write file .cache/uv/wheels-v1/index/b2a7eb67d4c26b82/attrs/attrs-24.2.0-py3-none-any","time":"2024-11-02T09:17:41Z"}
{"error":".zip archives do not support non-regular files","level":"error","msg":"unable to write file .cache/uv/wheels-v1/index/b2a7eb67d4c26b82/blinker/blinker-1.8.2-py3-none-any","time":"2024-11-02T09:17:41Z"}
{"error":".zip archives do not support non-regular files","level":"error","msg":"unable to write file .cache/uv/wheels-v1/index/b2a7eb67d4c26b82/cachetools/cachetools-5.5.0-py3-none-any","time":"2024-11-02T09:17:41Z"}
```

### .streamlit/config.toml

```plaintext
[server]
headless = true
address = "0.0.0.0"
port = 5000

[theme]
base = "dark"
primaryColor = "#9146FF"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
<<<<<<< HEAD
```
=======
```
```
>>>>>>> 569886ea8950f12bab31df00e6357f79f8ceebaa
