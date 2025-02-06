# Email Summarizer (On-Premise BART Model)

This project processes email data from a CSV file, cleans the email body, summarizes it using **Facebook's BART model**, and saves the summarized content to a new CSV file.

---

## 📌 Features
✅ **On-Premise Processing** – No external API calls, ensuring privacy.  
✅ **Uses BART for Summarization** – Optimized for text summarization tasks.  
✅ **Batch Processing** – Efficiently handles large email datasets.  
✅ **Customizable Parameters** – Configure model size, batch size, and output length.  

---

## 🚀 Installation

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/your-username/email_summarizer.git
cd email_summarization

python -m venv email-summrization
source email-summrization/bin/activate

pip install -r requirements.txt

python summarizer.py