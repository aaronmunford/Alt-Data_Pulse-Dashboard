# Alt-Data Pulse Dashboard

A quantitative signal platform for analyzing alternative data with an interactive Streamlit dashboard.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Quick Start

### Automated Setup (Recommended)

For macOS/Linux users, we provide a setup script that automates the entire installation process:

```bash
git clone https://github.com/aaronmunford/Alt-Data_Pulse-Dashboard.git
cd Alt-Data_Pulse-Dashboard
./setup.sh
```

Then run the dashboard:

```bash
source .venv/bin/activate && cd dashboard && streamlit run app.py
```

### Manual Setup Instructions

#### 1. Clone the Repository

```bash
git clone https://github.com/aaronmunford/Alt-Data_Pulse-Dashboard.git
cd Alt-Data_Pulse-Dashboard
```

#### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
```

#### 3. Activate the Virtual Environment

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

#### 4. Install Dependencies

**Option A: Install from requirements.txt (Recommended)**

```bash
pip install -r requirements.txt
```

**Option B: Install manually**

```bash
pip install streamlit plotly altair pandas numpy scipy statsmodels
```

**Note:** If you need data engineering capabilities, see `requirements-dataeng.txt` for optional dependencies.

## Running the Dashboard

After installing the dependencies, run the Streamlit application:

```bash
cd dashboard
streamlit run app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

## Project Structure

```
.
├── dashboard/          # Streamlit dashboard application
│   └── app.py         # Main dashboard file
├── data/              # Data storage
├── ingest/            # Data ingestion scripts
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Troubleshooting

### ModuleNotFoundError: No module named 'streamlit'

If you encounter this error:

```
ModuleNotFoundError: No module named 'streamlit.cli'
```

Make sure you:
1. Have activated your virtual environment
2. Have installed streamlit: `pip install streamlit`
3. Are running the command from the correct directory

### Missing Dependencies

If you encounter missing module errors, install the required packages:

```bash
pip install <package-name>
```

## License

See the repository for license information.
