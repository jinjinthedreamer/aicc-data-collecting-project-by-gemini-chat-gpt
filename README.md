# aicc-data-collecting-project-by-gemini-chat-gpt
OCR &amp; Transcript‑Driven AICC Conversation Analytics Pipeline with GPT‑4o(with no hand coding)
# AICC Conversation Analytics Pipeline

This repository provides a fully automated pipeline that combines **OCR** (image screenshots) and **chat/call transcripts** to extract and compute key AICC (AI Contact Center) performance metrics using **OpenAI GPT‑4o**.

## Features

- **OCR → Text**  
  Extract raw dialog text from screenshot images (`.png`/`.jpg`) using `pytesseract`.

- **Journey Phase Analysis**  
  Segment each session into 90‑second “phases” and collect average customer sentiment (5‑level scale) over time.

- **LLM‑Powered KPI Extraction**  
  Invoke GPT‑4o to generate JSON containing 11 metrics per session, including issue category, agent evaluation, resolution status, sentiment, handover rate, error frequency, SLA compliance, and more.

- **DataFrame Output**  
  Organize the results into a `pandas` DataFrame for immediate dashboarding or downstream analysis.

## Repository Structure

├── ocr_images/ # Input screenshot images for OCR
├── API_KEY.env # OpenAI API key (environment variable)
├── requirements.txt # Python dependencies
├── aicc_analysis.py # Main analysis script
└── README.md # Project documentation


## Installation & Usage


1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/aicc-ocr-gpt.git
   cd aicc-ocr-gpt

2. Create a virtual environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate    # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

3. Configure your API key
echo OPENAI_API_KEY="YOUR API KEY" > API_KEY.env

4. Run the analysis on terminal
python aicc_analysis.py


#there's two python file's 
1) using hugging face
change the API KEYS

2) using gpt
change the API KEYS

-----

Sample Output
SessionID	Phase	IssueCategory	AgentRating	Resolved	Sentiment	Handover	…
session1	entry	Shipping	A	true	3	0	…
session2	phase1	SystemError	B	false	5	1	…


Contributing & License : MIT License
Contributions: Pull requests welcome!

Feel free to adjust the GitHub URL and script name as needed!

License: MIT License
