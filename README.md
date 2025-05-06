# VoxPop Personas

This project processes persona data and interacts with the OpenAI API to generate insights based on the provided personas.

## Project Structure
```
.
├── data/
│   └── voxpopai_clean_personas.csv
├── src/
│   └── main.py
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Activate the pyPOSSUM conda environment:
```bash
conda activate pyPOSSUM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
Create a `.env` file in the root directory and add:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage
Run the main script:
```bash
python src/main.py
```

## Development
This project follows PEP 8 coding standards and uses type hints for better code clarity. 