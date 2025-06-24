# SkillSight

SkillSight is a career insight platform that helps users plan their career growth. Once users indicate which roles they are interested in, it compares the user's resume to the skills required for those roles and recommends jobs, online courses, and other next steps to help them close the gap.

The goal is to provide personalized, data-driven guidance to help users confidently progress toward their career goals.

## 🚀 Tech Stack

- **Backend:** Python, FastAPI, SQLite, SQLAlchemy
- **Frontend:** React.js, HTML, CSS

---

## 🛠️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/skillsight.git
cd skillsight
```
### 2. Create a virtual environment

```bash
python3 -m venv skillsight-env
source skillsight-env/bin/activate  # on macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r backend/requirements.txt
```

### 4. Run the backend (once built)
```bash
uvicorn main:app --reload
```

---