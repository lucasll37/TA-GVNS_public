# -----------------------------------------------------------------------------
# Makefile for Environment Setup, Running Scripts, and Cleaning Up
#
# This Makefile provides targets for creating a conda environment, installing
# system dependencies, running various Python scripts, creating videos,
# and cleaning build files, caches, and GPU memory.
# -----------------------------------------------------------------------------

.PHONY: create_env apt_deps run analysis video clean help

.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# Variables
# -----------------------------------------------------------------------------
VENV_DIR       := ./.venv
REQUIREMENTS   := requirements.txt
PYTHON         := python -u
# -----------------------------------------------------------------------------

# apt_deps:
# 	@echo "Installing system dependencies..."
# 	sudo apt-get update && sudo apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-dri mesa-utils ccache
# 	@bash -c "conda install libpython-static -y"

create_env:
	@echo "Creating conda environment in $(VENV_DIR)..."
	conda create --prefix $(VENV_DIR) python=3.9 -y
	@echo "Activating environment and installing dependencies..."
	@bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(VENV_DIR) && pip install --upgrade pip && pip install -r $(REQUIREMENTS) && echo 'Done.'"

run:
	@echo "Running the main Python script..."
	@bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(VENV_DIR) && $(PYTHON) ./src/main.py"

analysis:
	@echo "Running the analysis script..."
	@bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(VENV_DIR) && $(PYTHON) ./src/analysis.py"

video:
	@echo "Running the video creation script..."
	@bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(VENV_DIR) && $(PYTHON) ./utils/create_video.py"

clean:
	@echo "Cleaning build files and caches..."
	rm -rf $(VENV_DIR) ./.pytest_cache $(TA_GVNS_EXE) $(BUILD_DIR)
	find . -type f -name '*.py[co]' -delete
	find . -type d -name __pycache__ -delete
	@echo "Clean complete."

help:
	@echo "Makefile for Environment Setup, Running Scripts, and Cleaning Up"
	@echo ""
	@echo "Usage:"
	@echo "  make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  create_env   Create conda environment and install dependencies"
	@echo "  apt_deps     Install system dependencies (commented out)"
	@echo "  run          Run the main Python script"
	@echo "  analysis     Run the analysis script"
	@echo "  video        Run the video creation script"
	@echo "  clean        Clean build files and caches"