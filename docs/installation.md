# Installation Guide

This guide provides detailed instructions for installing and setting up the Stock Return Analyzer project.

## System Requirements

### Operating Systems
- **macOS** 10.15 or later (recommended)
- **Linux** (Ubuntu 20.04+, Fedora, CentOS, etc.)
- **Windows** 10 or later (with Python 3.13+)

### Software Requirements
- **Python 3.13 or higher**: The project requires Python 3.13+
- **pip**: Python package installer (usually comes with Python)
- **Git**: For cloning the repository (optional if downloading ZIP)

### Disk Space
- Minimum: 100 MB free space
- Recommended: 500 MB for dependencies and cached data

## Installation Methods

### Method 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver that provides better performance and dependency management.

1. **Install uv** (if not already installed):
   ```bash
   pip install uv
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/Yamiyorunoshura/project-part-A.git
   cd project-part-A
   ```

3. **Sync dependencies**:
   ```bash
   uv sync
   ```

   This command:
   - Creates a virtual environment (if not exists)
   - Installs all dependencies from `pyproject.toml`
   - Sets up the development environment

4. **Activate the virtual environment** (optional):
   ```bash
   source .venv/bin/activate  # Linux/macOS
   # or
   .venv\Scripts\activate     # Windows
   ```

### Method 2: Using pip

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Yamiyorunoshura/project-part-A.git
   cd project-part-A
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # or
   .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` doesn't exist, install from `pyproject.toml`:
   ```bash
   pip install -e .
   ```

### Method 3: Download ZIP (No Git)

1. **Download the repository**:
   - Go to https://github.com/Yamiyorunoshura/project-part-A
   - Click "Code" → "Download ZIP"
   - Extract the ZIP file to your preferred location

2. **Open terminal in the extracted folder**:
   ```bash
   cd /path/to/project-part-A
   ```

3. **Follow either uv or pip installation steps above**

## Verification

After installation, verify everything works:

1. **Check Python version**:
   ```bash
   python --version
   ```
   Should show `Python 3.13.x` or higher.

2. **Test imports**:
   ```bash
   python -c "import pandas; import yfinance; import matplotlib; print('All imports successful')"
   ```

3. **Run a quick test**:
   ```bash
   python -c "from main import StockReturnCalculator; print('StockReturnCalculator imported successfully')"
   ```

## Troubleshooting

### Common Issues

#### 1. Python Version Too Old
**Error**: `requires-python = ">=3.13"`

**Solution**:
- Upgrade Python to 3.13 or later
- Or modify `pyproject.toml` to accept older versions (not recommended)

#### 2. uv Command Not Found
**Error**: `uv: command not found`

**Solution**:
- Ensure uv is installed: `pip install uv`
- Check PATH: `which uv` (Linux/macOS) or `where uv` (Windows)
- Reinstall if necessary: `pip install --force-reinstall uv`

#### 3. Permission Denied Errors
**Error**: `PermissionError: [Errno 13]`

**Solution**:
- Use virtual environments (recommended)
- Don't install packages with system Python
- Use `--user` flag with pip if needed: `pip install --user uv`

#### 4. Network Issues (Behind Proxy)
**Error**: `ConnectionError` or timeout

**Solution**:
- Configure proxy settings for pip/uv
- Use mirror repositories if available
- Try offline installation

#### 5. Missing Dependencies on Linux
**Error**: `ImportError` for system libraries

**Solution** (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install python3-dev build-essential
```

**Solution** (Fedora/RHEL):
```bash
sudo dnf install python3-devel gcc-c++
```

### Virtual Environment Issues

#### Virtual Environment Not Activating
**Symptoms**: Commands run with system Python instead of virtual environment

**Solution**:
- Verify activation: `which python` should point to `.venv/bin/python`
- Manual activation:
  ```bash
  source .venv/bin/activate  # Linux/macOS
  .venv\Scripts\activate     # Windows
  ```

#### Cross-Platform Path Issues
**Windows-specific**: Use backslashes and correct script paths
```powershell
# PowerShell
.\.venv\Scripts\Activate.ps1

# Command Prompt
.venv\Scripts\activate.bat
```

## Development Setup

For contributors and developers:

1. **Install development dependencies**:
   ```bash
   uv sync --dev
   ```

2. **Set up pre-commit hooks** (optional):
   ```bash
   uv run pre-commit install
   ```

3. **Run code quality checks**:
   ```bash
   uv run ruff check . --fix
   uv run black .
   ```

## Updating Dependencies

To update to the latest versions:

```bash
# Using uv
uv sync --upgrade

# Using pip
pip install --upgrade -r requirements.txt
```

## Uninstallation

To completely remove the project:

1. **Delete the virtual environment**:
   ```bash
   rm -rf .venv
   ```

2. **Delete the project directory**:
   ```bash
   cd ..
   rm -rf project-part-A
   ```

## Next Steps

After successful installation:

1. **Run the main analysis**: `uv run main.py`
2. **Explore the code**: Read `main.py` to understand the implementation
3. **Try the sample**: Run `uv run sample.py` for additional examples
4. **Read the documentation**: Continue to [Usage Guide](usage.md)

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/Yamiyorunoshura/project-part-A/issues)
2. Review error messages carefully
3. Ensure all prerequisites are met
4. Consider creating a new issue with:
   - Your operating system and Python version
   - Complete error message
   - Steps to reproduce the issue