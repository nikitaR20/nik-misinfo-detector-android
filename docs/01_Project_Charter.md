# Project Charter

**Project Name:** Misinformation Detector – Real-Time Credibility Check App  
**Date:** September 27, 2025  
**Project Sponsor:** [Your Name / Organization]  
**Project Manager:** [Your Name]  
**Version:** 1.1  

---

## 1. Project Purpose / Justification
The proliferation of misinformation on social media poses risks to public knowledge and decision-making. Users often lack tools to verify content credibility in real time.

**Purpose:**  
- Develop an Android mobile app that allows users to assess the credibility of text or screen content quickly.  
- Provide a **probability score**, flagged phrases, and visual indicators (color-coded verdict).  
- Empower the general public to make informed decisions while browsing social media.

**Business Case / Benefits:**  
- Reduces the spread of false information.  
- Enhances digital literacy and awareness among users.  
- Provides a market-ready mobile solution in the growing AI content verification space.  
- Opportunities for partnerships with social media platforms and fact-checking organizations.

---

## 2. Project Objectives
1. Deliver a functional Android app with a **floating overlay icon** for real-time content checking.  
2. Integrate **OCR** for screenshots and AI/ML models for misinformation detection.  
3. Optionally integrate **fact-checking APIs** to validate information.  
4. Ensure **user-friendly interface** with clear verdict, flagged phrases, and history tracking.  
5. Maintain **secure, privacy-compliant handling** of user data.

---

## 3. Scope

### In Scope
- Android app development using Kotlin + Jetpack Compose.  
- Screenshot capture and text input for content analysis.  
- OCR integration (Google ML Kit) for text extraction.  
- Backend API (FastAPI + Python) for ML inference.  
- Display results in floating overlay popup (probability score, verdict, flagged phrases).  
- Local storage of history (SQLite).  
- Optional: sharing feature and fact-check API integration.

### Out of Scope
- iOS version during MVP.  
- Multi-language support initially.  
- Large-scale analytics dashboards for backend.

---

## 4. Deliverables
- Project Charter & Proposal Document.  
- Wireframes / UI mockups.  
- Kotlin Android App with overlay icon, OCR, UI.  
- FastAPI backend with deployed ML model.  
- Test Plan and executed test cases.  
- Deployment guide & user guide.  
- **GitHub Project Board** setup for tracking tasks, documents, and milestones.

---

## 5. Milestones / Timeline
| Milestone | Target Date | Description |
|-----------|------------|-------------|
| Project Kickoff | Week 0 | Confirm scope, team roles, tools |
| UI/UX Prototype | Week 2 | Floating icon, basic app layout |
| OCR Integration | Week 3 | Screenshot capture & text extraction |
| ML Model Training | Week 4 | Fine-tune and test AI/ML model |
| Backend Deployment | Week 5 | FastAPI server with ML inference |
| Frontend-Backend Integration | Week 6 | Connect Android app to backend API |
| Features Complete | Week 7 | Result popup, history, sharing |
| Testing & Documentation | Week 8 | QA, bug fixes, user guide |

---

## 6. Stakeholders
| Stakeholder | Role |
|------------|------|
| Project Sponsor | Provides funding, approval |
| Project Manager | Oversees project, timelines, resources |
| Business Analyst | Gathers requirements, defines user stories |
| Project Architect | Defines technical architecture, integration plan |
| Developers | Implement Kotlin frontend & FastAPI backend |
| QA / Testers | Validate functionality, performance, accuracy |
| End Users | Android users seeking misinformation detection |

---

## 7. Assumptions
- Users will grant permissions for floating overlay and screen capture.  
- Internet connectivity is available for backend ML inference (optional offline ML for MVP).  
- Social media content text can be reliably extracted via OCR.  
- ML model accuracy is sufficient (target ≥85% confidence).  

---

## 8. Constraints
- App limited to **Android devices** initially.  
- Development limited to **8-week MVP timeline**.  
- Backend hosting resources constrained by free-tier platforms (Railway/Heroku).  
- Privacy and data protection laws must be followed (GDPR/CCPA compliant).  

---

## 9. Risks
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Floating overlay may not work on all Android versions | Medium | High | Test on multiple devices, fallback to in-app capture |
| ML model too large for device | High | Medium | Use lightweight on-device model or server-side inference |
| API downtime | Medium | Medium | Cache responses, fallback to local ML inference |
| User privacy concerns | Medium | High | Secure data storage, privacy policy, minimal local storage |

---

## 10. Technology / Tools
- **Frontend:** Kotlin + Jetpack Compose  
- **Backend:** FastAPI + Python + Hugging Face Transformers  
- **OCR:** Google ML Kit  
- **Database:** SQLite (local history)  
- **Networking:** Retrofit2  
- **Project Management / Tracking:** GitHub Projects (Boards, Issues, Milestones)  
- **Design / Wireframes:** Figma, Draw.io  
- **Version Control:** Git + GitHub Repos  
- **Hosting:** Railway / Heroku (MVP), AWS/GCP (Production)  

---

## 11. Approval Signatures
| Role | Name | Signature | Date |
|------|------|----------|------|
| Project Sponsor |  |  |  |
| Project Manager |  |  |  |
| Business Analyst |  |  |  |
| Project Architect |  |  |  |
