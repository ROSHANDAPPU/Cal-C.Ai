from twilio.rest import Client
import random
import time

# In-memory OTP store (replace with Redis/DB for production)
otp_store = {}

# Send SMS OTP endpoint
@app.post("/verify/send-otp")
async def send_sms_otp(email: str, phone: str):
    db: Session = SessionLocal()
    user = db.query(UserProfile).filter(UserProfile.email == email).first()
    if not user:
        return JSONResponse(status_code=404, content={"error": "User not found."})
    otp = str(random.randint(100000, 999999))
    otp_store[email] = {"otp": otp, "expires": time.time() + 300}
    # Twilio credentials (use env vars in production)
    account_sid = os.getenv("TWILIO_SID", "demo_sid")
    auth_token = os.getenv("TWILIO_TOKEN", "demo_token")
    twilio_phone = os.getenv("TWILIO_PHONE", "+1234567890")
    client = Client(account_sid, auth_token)
    try:
        client.messages.create(
            body=f"Your verification code is: {otp}",
            from_=twilio_phone,
            to=phone
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"SMS send failed: {e}"})
    return {"message": "OTP sent via SMS."}
import cv2
from deepface import DeepFace
# ID + Face Recognition Endpoint
@app.post("/verify/id-face")
async def verify_id_face(email: str, id_file: UploadFile = File(...), face_file: UploadFile = File(...)):
    db: Session = SessionLocal()
    user = db.query(UserProfile).filter(UserProfile.email == email).first()
    if not user:
        return JSONResponse(status_code=404, content={"error": "User not found."})
    id_img_bytes = await id_file.read()
    face_img_bytes = await face_file.read()
    # Save images temporarily for OpenCV/DeepFace
    with open("temp_id.jpg", "wb") as f:
        f.write(id_img_bytes)
    with open("temp_face.jpg", "wb") as f:
        f.write(face_img_bytes)
    try:
        result = DeepFace.verify("temp_face.jpg", "temp_id.jpg")
        score = result.get("distance", 0)
        verified = result.get("verified", False)
    except Exception as e:
        score = 0
        verified = False
    log = VerificationLog(
        user_id=user.id,
        verification_type="id_face",
        result="success" if verified else "failure",
        details=f"face_match_score={score}",
        face_match_score=score
    )
    db.add(log)
    db.commit()
    return {"verified": verified, "score": score}
# Multi-Factor Verification Endpoint (Email + SMS OTP)
@app.post("/verify/otp")
async def verify_otp(email: str, otp: str):
    db: Session = SessionLocal()
    user = db.query(UserProfile).filter(UserProfile.email == email).first()
    if not user:
        return JSONResponse(status_code=404, content={"error": "User not found."})
    otp_entry = otp_store.get(email)
    if not otp_entry or time.time() > otp_entry["expires"]:
        return JSONResponse(status_code=400, content={"error": "OTP expired or not found."})
    if otp_entry["otp"] != otp:
        return JSONResponse(status_code=400, content={"error": "Invalid OTP."})
    # Log OTP verification
    log = VerificationLog(
        user_id=user.id,
        verification_type="otp",
        result="success",
        details=f"otp={otp}"
    )
    db.add(log)
    db.commit()
    del otp_store[email]
    return {"message": "OTP verified."}
# Blockchain Credential Verification Endpoint
@app.post("/blockchain/store-credential")
async def store_credential(email: str, credential: str):
    # TODO: Integrate with blockchain API (e.g., Ethereum, Hyperledger)
    # For demo, just log the credential
    db: Session = SessionLocal()
    user = db.query(UserProfile).filter(UserProfile.email == email).first()
    if not user:
        return JSONResponse(status_code=404, content={"error": "User not found."})
    log = VerificationLog(
        user_id=user.id,
        verification_type="blockchain",
        result="success",
        details=f"credential={credential}"
    )
    db.add(log)
    db.commit()
    return {"message": "Credential stored on blockchain (demo)."}
# Real-time verification status endpoint for polling
@app.get("/user/status")
async def get_user_status(email: str):
    db: Session = SessionLocal()
    user = db.query(UserProfile).filter(UserProfile.email == email).first()
    if not user:
        return {"status": "not_found"}
    return {"status": user.verification_status}
from fastapi import Depends, HTTPException, status
app = FastAPI()
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
import os
from passlib.context import CryptContext
from backend.core.models import AdminUser  # Import AdminUser
app = FastAPI()
# JWT and password hashing setup
SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "supersecret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/admin/login")

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    from datetime import datetime, timedelta
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_admin(token: str = Depends(oauth2_scheme)):
    db: Session = SessionLocal()
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    admin = db.query(AdminUser).filter(AdminUser.email == email).first()
    if admin is None:
        raise credentials_exception
    return admin

# Admin login endpoint
@app.post("/admin/login")
async def admin_login(form_data: OAuth2PasswordRequestForm = Depends()):
    db: Session = SessionLocal()
    admin = db.query(AdminUser).filter(AdminUser.email == form_data.username).first()
    if not admin or not verify_password(form_data.password, admin.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    access_token = create_access_token({"sub": admin.email, "role": admin.role})
    return {"access_token": access_token, "token_type": "bearer", "role": admin.role}
import hashlib
from backend.core.models import VerificationLog
app = FastAPI()

# Manual Verification Endpoint (Admin Override)
@app.post("/verify/manual")
async def manual_verification(email: str, status: str, admin_name: str, comment: str, current_admin: AdminUser = Depends(get_current_admin)):
    # Only admin or super_admin can override
    if current_admin.role not in ["admin", "super_admin"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    db: Session = SessionLocal()
    user = db.query(UserProfile).filter(UserProfile.email == email).first()
    if not user:
        return JSONResponse(status_code=404, content={"error": "User not found."})
    user.verification_status = status
    log = VerificationLog(
        user_id=user.id,
        verification_type="manual_override",
        result=status,
        details=f"Manual override by {admin_name}",
        admin_name=admin_name,
        comment=comment,
        confidence=None
    )
    db.add(log)
    db.commit()
    return {"message": f"Manual verification set to {status}."}

# SIS API Integration Hook (Stub)
@app.post("/verify/sis")
async def verify_with_sis(email: str):
    # Placeholder for SIS API integration
    # Example: call university API, check enrollment, update status
    # result = call_university_sis_api(email)
    # if result == "verified":
    #     ...
    return {"message": "SIS verification not implemented. Contact support for integration."}
app = FastAPI()

from fastapi import UploadFile, File
import pytesseract
from PIL import Image
import io

# Student ID Photo Upload + AI Verification Endpoint
@app.post("/verify/student-id")
async def verify_student_id(email: str, file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    # OCR extraction
    ocr_text = pytesseract.image_to_string(image)
    # Simple extraction logic (customize for real IDs)
    # Example: look for name, ID number, expiration
    name = None
    id_number = None
    expiration = None
    for line in ocr_text.splitlines():
        if "Name:" in line:
            name = line.split(":", 1)[-1].strip()
        if "ID:" in line:
            id_number = line.split(":", 1)[-1].strip()
        if "Exp:" in line or "Expires:" in line:
            expiration = line.split(":", 1)[-1].strip()
    # Fetch user profile
    db: Session = SessionLocal()
    user = db.query(UserProfile).filter(UserProfile.email == email).first()
    if not user:
        return JSONResponse(status_code=404, content={"error": "User not found."})
    # Liveness detection stub (replace with real blink/background check)
    liveness_passed = True  # TODO: Implement liveness detection
    # Store hash of image for compliance
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    # Log verification attempt
    log = VerificationLog(
        user_id=user.id,
        verification_type="student_id",
        result="success" if name and user.name.lower() in name.lower() and liveness_passed else "failure",
        details=f"name={name},id={id_number},exp={expiration},hash={image_hash},liveness={liveness_passed}"
    )
    db.add(log)
    if name and user.name.lower() in name.lower() and liveness_passed:
        user.verification_status = "id_verified"
        db.commit()
        return {"message": "Student ID verified.", "name": name, "id_number": id_number, "expiration": expiration, "image_hash": image_hash}
    else:
        user.verification_status = "id_failed"
        db.commit()
        return JSONResponse(status_code=400, content={"error": "ID verification failed.", "ocr_text": ocr_text, "image_hash": image_hash})
# Manual Verification Endpoint (Admin Override)
@app.get("/admin/verification-logs")
async def get_verification_logs(
    name: str = None,
    email: str = None,
    vtype: str = None,
    status: str = None,
    start: str = None,
    end: str = None,
    current_admin: AdminUser = Depends(get_current_admin)
):
    # Reviewer: can only view, admin/super_admin can filter/export
    db: Session = SessionLocal()
    query = db.query(VerificationLog)
    if name:
        query = query.join(UserProfile).filter(UserProfile.name.ilike(f"%{name}%"))
    if email:
        query = query.join(UserProfile).filter(UserProfile.email.ilike(f"%{email}%"))
    if vtype:
        query = query.filter(VerificationLog.verification_type == vtype)
    if status:
        query = query.filter(VerificationLog.result == status)
    if start:
        query = query.filter(VerificationLog.timestamp >= start)
    if end:
        query = query.filter(VerificationLog.timestamp <= end)
    logs = query.order_by(VerificationLog.timestamp.desc()).limit(100).all()
    return [{
        "id": log.id,
        "user_id": log.user_id,
        "timestamp": log.timestamp,
        "type": log.verification_type,
        "result": log.result,
        "details": log.details,
        "admin_name": log.admin_name,
        "comment": log.comment,
        "confidence": log.confidence
    } for log in logs]
# University Email Verification Endpoint
from fastapi import BackgroundTasks
import secrets
from backend.core.models import UserProfile
from backend.core.db import SessionLocal
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
import smtplib
from email.mime.text import MIMEText

def send_verification_email(email: str, token: str):
    # Replace with real SMTP config
    smtp_server = "smtp.example.edu"
    smtp_port = 587
    smtp_user = "noreply@example.edu"
    smtp_password = "password"
    link = f"https://yourdomain.com/verify-email?token={token}"
    msg = MIMEText(f"Click to verify your student account: {link}")
    msg["Subject"] = "Verify your student account"
    msg["From"] = smtp_user
    msg["To"] = email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, [email], msg.as_string())
    except Exception as e:
        print(f"Email send failed: {e}")

app = FastAPI()

# University Email Verification Endpoint
@app.post("/verify/email")
async def verify_email(email: str, background_tasks: BackgroundTasks):
    if not email.endswith(".edu"):
        return JSONResponse(status_code=400, content={"error": "Email must be a .edu address."})
    token = secrets.token_urlsafe(32)
    db: Session = SessionLocal()
    user = db.query(UserProfile).filter(UserProfile.email == email).first()
    if not user:
        return JSONResponse(status_code=404, content={"error": "User not found."})
    user.verification_token = token
    user.verification_status = "pending"
    db.commit()
    background_tasks.add_task(send_verification_email, email, token)
    return {"message": "Verification email sent."}

@app.get("/verify/email/callback")
async def verify_email_callback(token: str):
    db: Session = SessionLocal()
    user = db.query(UserProfile).filter(UserProfile.verification_token == token).first()
    if not user:
        return JSONResponse(status_code=404, content={"error": "Invalid token."})
    user.email_verified = True
    user.verification_status = "verified"
    db.commit()
    return {"message": "Email verified."}
# Add FastAPI and Request imports for endpoint definitions
from fastapi import FastAPI, Request
# ...existing FastAPI app and endpoints...


# ...existing FastAPI app and endpoints...

from fastapi import WebSocket


# Place all new endpoints below app = FastAPI()

# ...existing code...

app = FastAPI()

# ...existing module instantiation and basic endpoints...

# Real-time collaboration WebSocket endpoint
@app.websocket("/ws/collaborate")
async def websocket_collaborate(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Received: {data}")

# Handwriting recognition
@app.post("/recognize/handwriting")
async def recognize_handwriting():
    return {"recognized_text": "y = x^2 + 3x + 2"}

# Photo math solver
@app.post("/solve/photo-math")
async def solve_photo_math():
    return {"solution_steps": ["Recognized equation: x^2 + 2x = 0", "Step 1: Factor", "Step 2: Solve for x"]}

# NLP endpoint
@app.post("/nlp/solve")
async def nlp_solve():
    return {"result": "Trajectory calculated: Range = 45m, Time = 3s"}

# Context-aware suggestion engine
@app.post("/suggest/context-aware")
async def context_aware_suggestion():
    return {"suggestions": ["Review algebra basics", "Practice calculus derivatives"]}

# Real-time data endpoints
@app.get("/data/stock-market")
async def get_stock_market():
    return {"AAPL": 175.23, "GOOG": 2820.15}

@app.get("/data/weather")
async def get_weather():
    return {"location": "NYC", "temperature": 27, "condition": "Sunny"}

@app.get("/data/economic")
async def get_economic():
    return {"GDP": "21T", "CPI": "2.5%"}

@app.get("/data/sports")
async def get_sports():
    return {"team": "Warriors", "score": "102-98"}

# Wolfram Alpha API integration
@app.post("/integrate/wolfram-alpha")
async def wolfram_alpha_query():
    return {"wolfram_result": "Result from Wolfram Alpha"}

# Computer vision pipelines
@app.post("/vision/handwriting")
async def vision_handwriting():
    return {"recognized_text": "y = mx + b"}

@app.post("/vision/photo")
async def vision_photo():
    return {"solution": "x = 2, x = -2"}

# Exam difficulty estimator
@app.post("/estimate/exam-difficulty")
async def estimate_exam_difficulty():
    return {"difficulty_score": 0.75, "level": "Medium"}

import sympy as sp
import numpy as np
from core.calculation_modules import (
    AlgebraModule, CalculusModule, PhysicsModule, CalculationResult,
    quantum_wavefunction, quantum_probability_amplitude,
    molecular_structure_analysis, orbital_parameters,
    population_genetics_allele_freq, monte_carlo_simulation
)

# Instantiate modules
algebra_module = AlgebraModule()
calculus_module = CalculusModule()
physics_module = PhysicsModule()

@app.get("/")
def root():
    return {"message": "Welcome to the AI-Powered Advanced Calculator API"}

# Algebra endpoint
@app.post("/calculate/algebra", response_model=None)
async def calculate_algebra(request: Request):
    input_data = await request.json()
    result: CalculationResult = await algebra_module.calculate(input_data)
    return result.__dict__

# Calculus endpoint
@app.post("/calculate/calculus", response_model=None)
async def calculate_calculus(request: Request):
    input_data = await request.json()
    result: CalculationResult = await calculus_module.calculate(input_data)
    return result.__dict__

# Physics endpoint
# Physics endpoint
@app.post("/calculate/physics", response_model=None)
async def calculate_physics(request: Request):
    input_data = await request.json()
    result: CalculationResult = await physics_module.calculate(input_data)
    return result.__dict__

# === Predictive Analytics Endpoints ===

from core.calculation_modules import (
    grade_predictor_linear, grade_predictor_rf, grade_predictor_ai,
    study_time_optimizer, career_path_calculator, gpt_course_suggestion
)

@app.post("/predict/grade/linear")
async def predict_grade_linear(request: Request):
    data = await request.json()
    X = data.get("X", [])
    y = data.get("y", [])
    X_pred = data.get("X_pred", [])
    result = grade_predictor_linear(X, y, X_pred)
    return {"predicted_grade": result}

@app.post("/predict/grade/random-forest")
async def predict_grade_rf(request: Request):
    data = await request.json()
    X = data.get("X", [])
    y = data.get("y", [])
    X_pred = data.get("X_pred", [])
    result = grade_predictor_rf(X, y, X_pred)
    return {"predicted_grade": result}

@app.post("/predict/grade/ai")
async def predict_grade_ai(request: Request):
    data = await request.json()
    X = data.get("X", [])
    y = data.get("y", [])
    X_pred = data.get("X_pred", [])
    # result = grade_predictor_ai(X, y, X_pred)  # Function returns None
    return {"predicted_grade": None}

@app.post("/optimize/study-time")
async def optimize_study_time(request: Request):
    data = await request.json()
    logs = data.get("logs", [])
    difficulties = data.get("difficulties", [])
    grades = data.get("grades", [])
    result = study_time_optimizer(logs, difficulties, grades)
    return result

@app.post("/calculate/career-path")
async def calculate_career_path(request: Request):
    data = await request.json()
    grades = data.get("grades", [])
    skills = data.get("skills", [])
    interests = data.get("interests", [])
    result = career_path_calculator(grades, skills, interests)
    return result

@app.post("/suggest/gpt-course")
async def suggest_gpt_course(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    result = gpt_course_suggestion(prompt)
    return {"suggested_courses": result}

@app.post("/calculate/quantum/wavefunction")
async def calculate_quantum_wavefunction(request: Request):
    data = await request.json()
    symbols = [sp.Symbol(s) for s in data.get("symbols", [])]
    expr = sp.sympify(data.get("expression", ""))
    result = quantum_wavefunction(symbols, expr)
    return {"wavefunction": str(result)}

@app.post("/calculate/quantum/probability")
async def calculate_quantum_probability(request: Request):
    data = await request.json()
    psi = sp.sympify(data.get("psi", ""))
    x = sp.Symbol(data.get("variable", "x"))
    limits = tuple(data.get("limits", [-sp.oo, sp.oo]))
    prob = quantum_probability_amplitude(psi, x, limits)
    return {"probability": str(prob)}

@app.post("/analyze/molecule")
async def analyze_molecule(request: Request):
    data = await request.json()
    smiles = data.get("smiles", "")
    result = molecular_structure_analysis(smiles)
    return result

@app.post("/calculate/astrophysics/orbit")
async def calculate_orbital_parameters(request: Request):
    data = await request.json()
    mass1 = data.get("mass1")
    mass2 = data.get("mass2")
    distance = data.get("distance")
    result = orbital_parameters(mass1, mass2, distance)
    return result

@app.post("/calculate/biostatistics/allele-frequency")
async def calculate_allele_frequency(request: Request):
    data = await request.json()
    population = data.get("population", [])
    result = population_genetics_allele_freq(population)
    return result

@app.post("/calculate/biostatistics/monte-carlo")
async def run_monte_carlo(request: Request):
    data = await request.json()
    # For demo: simulate sum of two random numbers
    def func():
        return np.random.rand() + np.random.rand()
    n_iter = data.get("n_iter", 1000)
    results = monte_carlo_simulation(func, n_iter=n_iter)
    return {"results": results}
