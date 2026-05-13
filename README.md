1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ats-resume-selector.git
cd ats-resume-selector
2. Install dependencies
pip install -r requirements.txt
3. Run the app
streamlit run ats_app.py
4. Open in browser
http://localhost:8501
📊 How It Works

Extracts text from PDF/DOCX resumes
Cleans and tokenizes text using NLTK
Computes keyword match score (40% weight)
Computes semantic similarity using Hugging Face all-MiniLM-L6-v2 (60% weight)
Combines scores and ranks all candidates

👨‍💻 Author
Rohan — MSc Artificial Intelligence, Dublin City University