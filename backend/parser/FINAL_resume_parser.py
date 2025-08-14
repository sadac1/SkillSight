import re
import os
import spacy
import json
from spacy.matcher import Matcher, PhraseMatcher
from rapidfuzz import fuzz
from pypdf import PdfReader
from docx import Document
from collections import Counter
from string import punctuation
from typing import Dict, List, Tuple, Optional

class final_test_parser:
    def __init__(self, file):
        self.file = file
        self.text = self.get_text()
        self.nlp = spacy.load("en_core_web_sm")
        self.final_text = ""
        self.skills_dict = {}
        self.matchers = {}

    def get_text(self):
        if self.file.lower().endswith('.pdf'):
            return self.PDF_get_text()
        elif self.file.lower().endswith('.docx'):
            return self.DOCX_get_text()
        else:
            raise ValueError("Unsupported file format")


    def PDF_get_text(self):
        try:
            reader = PdfReader(self.file)
            page = reader.pages[0]
            text = page.extract_text()
            return text if text else ""
        except FileNotFoundError:
            raise FileNotFoundError("File not found")
        except Exception as e:
            raise RuntimeError("Failed to read PDF")

    def DOCX_get_text(self):
        try:
            print("Loaded Succesfully")
            document = Document(self.file)
            full_text = []
            for para in document.paragraphs:
                full_text.append(para.text)
            return '\n'.join(full_text)
        except FileNotFoundError:
            raise FileNotFoundError("File not found")
        except Exception as e:
            raise RuntimeError("Failed to read DOCX")

    #cleaning text to same format
    def clean_text(self):
        cleaned_raw = re.sub(r'[•·●]', ' ', self.text)
        cleaned_raw = re.sub(r'\s+', ' ', cleaned_raw).strip()

        text = self.nlp(cleaned_raw.lower())

        tokens = [
            token.lemma_ for token in text
            if not token.is_stop and not token.is_punct and not token.is_space
        ]

        self.final_text = ' '.join(tokens)
        
        #for skill matching
        self.skills_text = cleaned_raw.lower()

        return self.final_text
    

    def load_skills_from_file(self, path: str):
        with open(path, 'r') as f:
            self.skills_dict = json.load(f)
        self.build_phrase_matchers()

    def build_phrase_matchers(self):
        self.matchers = {}
        for category, skills in self.skills_dict.items():
            matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            patterns = [self.nlp(skill.lower()) for skill in skills]
            matcher.add(category, patterns)
            self.matchers[category] = matcher

    def extract_skills(self, fuzzy_threshold=90):
        # Class-level exact match categories
        EXACT_MATCH_CATEGORIES = {"programming_languages", "frameworks_libraries", "databases", "software_tools"}

        doc = self.nlp(self.skills_text)
        found_skills = {cat: set() for cat in self.skills_dict}

        print("\n=== EXACT MATCHING (PhraseMatcher) ===")
        for category, matcher in self.matchers.items():
            matches = matcher(doc)
            for match_id, start, end in matches:
                matched_text = doc[start:end].text
                print(f"[Exact] {category}: {matched_text}")
                found_skills[category].add(matched_text.lower())

        print("\n=== FUZZY MATCHING (RapidFuzz) ===")
        for category, skills in self.skills_dict.items():
            # Lowercase comparison for exact-match categories
            if category.lower() in EXACT_MATCH_CATEGORIES:
                continue  # skip fuzzy matching for exact-match categories

            for skill in skills:
                if skill not in found_skills[category]:
                    score = fuzz.partial_ratio(skill.lower(), self.skills_text)
                    if score >= fuzzy_threshold:
                        print(f"[Fuzzy {score}] {category}: {skill}")
                        found_skills[category].add(skill.lower())

        print("\n=== FINAL SKILLS BY CATEGORY ===")
        for category, skills in found_skills.items():
            display_skills = [s.title() for s in sorted(skills)]
            print(f"{category}: {', '.join(display_skills) if skills else 'None'}")

        return {cat: sorted(list(skills)) for cat, skills in found_skills.items()}


if __name__ == "__main__":
    resume_path = "/Users/ishanisahu/Desktop/GitHub/SkillSight/backend/parser/businessmarketingsampleresume.pdf"
    skills_path = "/Users/ishanisahu/Desktop/GitHub/SkillSight/backend/parser/skills.json"

    parser = final_test_parser(resume_path)
    parser.load_skills_from_file(skills_path)
    parser.clean_text()

    # Extract skills (with fuzzy matching)
    skills_found = parser.extract_skills(fuzzy_threshold=90)

    # Display final results
    print("\n=== Extracted Skills ===")
    for category, skills in skills_found.items():
        print(f"{category}: {', '.join(skills) if skills else 'None'}")
