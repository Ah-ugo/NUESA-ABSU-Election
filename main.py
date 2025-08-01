import logging
import os
import secrets
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import cloudinary
import cloudinary.uploader
import smtplib
from bson import ObjectId
from email.message import EmailMessage
from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field, field_serializer, field_validator
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="NUESA Voting System API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
RESET_TOKEN_EXPIRE_HOURS = 1
BASE_URL = os.getenv("BASE_URL", "https://nuesa-absu-election.onrender.com")

# SMTP Configuration
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = 465  # Hardcode to port 465 for SSL
SMTP_USERNAME = os.getenv("EMAIL_USER")
SMTP_PASSWORD = os.getenv("EMAIL_PASS")
SENDER_EMAIL = os.getenv("EMAIL_USER", SMTP_USERNAME)

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGODB_URL)
db = client.nuesa_voting

# Cloudinary configuration
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# Collections
users_collection = db.users
candidates_collection = db.candidates
elections_collection = db.elections
votes_collection = db.votes
departments_collection = db.departments
levels_collection = db.levels
announcements_collection = db.announcements
election_types_collection = db.election_types
positions_collection = db.positions
password_reset_tokens_collection = db.password_reset_tokens


# Pydantic Models
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        schema = handler(core_schema)
        schema.update(type="string")
        return schema


# User Models
class UserBase(BaseModel):
    firstName: str
    lastName: str
    email: str
    matricNumber: str
    department: str
    level: str


class UserCreate(UserBase):
    password: str
    profileImage: Optional[str] = None


class UserUpdate(BaseModel):
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    email: Optional[str] = None
    level: Optional[str] = None
    profileImage: Optional[str] = None


class UserInDB(UserBase):
    password: str
    profileImage: Optional[str] = None
    is_admin: bool = False
    created_at: datetime
    updated_at: datetime
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")

    @field_serializer('id')
    def serialize_id(self, id: PyObjectId, _info):
        return str(id)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class UserResponse(UserBase):
    id: str
    profileImage: Optional[str] = None
    is_admin: bool
    created_at: datetime
    updated_at: datetime


# Authentication Models
class AuthModel(BaseModel):
    matricNumber: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


# Election Models
class ElectionBase(BaseModel):
    title: str
    description: Optional[str] = None
    start_date: datetime
    end_date: datetime
    status: str = "pending"


class ElectionCreate(ElectionBase):
    election_type: str
    department: Optional[str] = None


class ElectionUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    status: Optional[str] = None
    election_type: Optional[str] = None
    department: Optional[str] = None


class ElectionInDB(ElectionBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    election_type: str
    department: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    created_by: str

    @field_serializer('id')
    def serialize_id(self, id: PyObjectId, _info):
        return str(id)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class ElectionResponse(ElectionBase):
    id: str
    election_type: str
    department: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    created_by: str


# Election Type Models
class ElectionTypeBase(BaseModel):
    key: str
    name: str
    description: Optional[str] = None


class ElectionTypeCreate(ElectionTypeBase):
    pass


class ElectionTypeUpdate(BaseModel):
    key: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None


class ElectionTypeInDB(ElectionTypeBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime
    updated_at: datetime
    created_by: str

    @field_serializer('id')
    def serialize_id(self, id: PyObjectId, _info):
        return str(id)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class ElectionTypeResponse(ElectionTypeBase):
    id: str
    created_at: datetime
    updated_at: datetime
    created_by: str


# Position Models
class PositionBase(BaseModel):
    name: str
    election_type: str
    department: Optional[str] = None


class PositionCreate(PositionBase):
    pass


class PositionUpdate(BaseModel):
    name: Optional[str] = None
    election_type: Optional[str] = None
    department: Optional[str] = None


class PositionInDB(PositionBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime
    updated_at: datetime
    created_by: str

    @field_serializer('id')
    def serialize_id(self, id: PyObjectId, _info):
        return str(id)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class PositionResponse(PositionBase):
    id: str
    created_at: datetime
    updated_at: datetime
    created_by: str


# Candidate Models
class CandidateBase(BaseModel):
    fullName: str
    position: str
    election_id: str
    election_type: str
    department: Optional[str] = None
    level: str
    manifesto: str


class CandidateCreate(CandidateBase):
    photo: Optional[str] = None


class CandidateInDB(CandidateBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    photo: Optional[str] = None
    vote_count: int = 0
    created_at: datetime
    updated_at: datetime
    created_by: str

    @field_serializer('id')
    def serialize_id(self, id: PyObjectId, _info):
        return str(id)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class CandidateResponse(CandidateBase):
    id: str
    photo: Optional[str] = None
    vote_count: int
    created_at: datetime
    updated_at: datetime
    created_by: str


# Vote Models
class VoteBase(BaseModel):
    candidate_id: str
    position: str
    election_id: str
    election_type: str
    department: Optional[str] = None

    @field_validator('candidate_id')
    def validate_candidate_id(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("'candidate_id' must be a valid ObjectId (24-character hex string)")
        return v

    @field_validator('election_id')
    def validate_election_id(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("'election_id' must be a valid ObjectId (24-character hex string)")
        return v


class VoteCreate(VoteBase):
    pass


class VoteInDB(VoteBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    created_at: datetime

    @field_serializer('id')
    def serialize_id(self, id: PyObjectId, _info):
        return str(id)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class BatchVoteCreate(BaseModel):
    votes: List[VoteCreate]


class BatchVoteResponse(BaseModel):
    message: str
    successful: int
    failed: int
    successful_votes: List[Dict]
    failed_votes: List[Dict]


class VotingStatusResponse(BaseModel):
    voted_positions: List[str]
    total_votes: int


# Announcement Models
class AnnouncementBase(BaseModel):
    title: str
    content: str
    priority: Optional[str] = "normal"


class AnnouncementCreate(AnnouncementBase):
    pass


class AnnouncementInDB(AnnouncementBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime
    created_by: str

    @field_serializer('id')
    def serialize_id(self, id: PyObjectId, _info):
        return str(id)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class AnnouncementResponse(AnnouncementBase):
    id: str
    created_at: datetime
    created_by: str


# Department Models
class DepartmentBase(BaseModel):
    name: str
    faculty: Optional[str] = None


class DepartmentCreate(DepartmentBase):
    pass


class DepartmentInDB(DepartmentBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime

    @field_serializer('id')
    def serialize_id(self, id: PyObjectId, _info):
        return str(id)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class DepartmentResponse(DepartmentBase):
    id: str
    created_at: datetime


# Level Models
class LevelBase(BaseModel):
    name: str


class LevelCreate(LevelBase):
    pass


class LevelInDB(LevelBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime

    @field_serializer('id')
    def serialize_id(self, id: PyObjectId, _info):
        return str(id)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class LevelResponse(LevelBase):
    id: str
    created_at: datetime


# Utility functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = await users_collection.find_one({"_id": ObjectId(user_id)})
    if user is None:
        raise credentials_exception
    return user


async def get_admin_user(current_user: dict = Depends(get_current_user)):
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


def upload_to_cloudinary(file_content, folder="nuesa_voting"):
    try:
        result = cloudinary.uploader.upload(
            file_content,
            folder=folder,
            resource_type="auto"
        )
        return result["secure_url"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image upload failed: {str(e)}")


def generate_reset_token():
    return secrets.token_urlsafe(32)


def send_reset_email(to_email: str, reset_url: str):
    msg = EmailMessage()
    msg.set_content("Please use the following link to reset your password: " + reset_url)  # Plain text fallback

    # Inline HTML email content
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Password Reset</title>
    </head>
    <body style="font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f4f4f4;">
        <div style="max-width: 600px; margin: 20px auto; background-color: #ffffff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h2 style="color: #333333; text-align: center;">Password Reset Request</h2>
            <p style="color: #555555; line-height: 1.6;">
                Dear User,
            </p>
            <p style="color: #555555; line-height: 1.6;">
                You have requested a password reset for your NUESA Voting System account. Please click the button below to reset your password:
            </p>
            <div style="text-align: center; margin: 20px 0;">
                <a href="{}" style="display: inline-block; padding: 10px 20px; background-color: #4f46e5; color: #ffffff; text-decoration: none; border-radius: 4px; font-weight: bold;">Reset Password</a>
            </div>
            <p style="color: #555555; line-height: 1.6;">
                This link will expire in 1 hour. If you did not request a password reset, please ignore this email or contact support.
            </p>
            <p style="color: #555555; line-height: 1.6;">
                Best regards,<br>
                NUESA Voting System Team
            </p>
            <hr style="border: none; border-top: 1px solid #eeeeee; margin: 20px 0;">
            <p style="color: #999999; font-size: 12px; text-align: center;">
                © 2025 NUESA Voting System. All rights reserved.
            </p>
        </div>
    </body>
    </html>
    """.format(reset_url)

    msg.add_alternative(html_content, subtype="html")

    msg["Subject"] = "Password Reset Request - NUESA Voting System"
    msg["From"] = SENDER_EMAIL
    msg["To"] = to_email

    try:
        logger.info(f"Attempting to send reset email to {to_email}")
        logger.info(
            f"SMTP settings: server={SMTP_SERVER}, port={SMTP_PORT}, username={SMTP_USERNAME}, password={'*' * len(SMTP_PASSWORD) if SMTP_PASSWORD else 'Not set'}")

        if not SMTP_USERNAME or not SMTP_PASSWORD:
            logger.error("SMTP_USERNAME or SMTP_PASSWORD is not set")
            raise HTTPException(status_code=500, detail="SMTP credentials not configured")

        # Use SMTP_SSL for port 465
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=30)
        logger.info(f"Connecting to {SMTP_SERVER}:{SMTP_PORT} with SSL")

        try:
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            logger.info("SMTP login successful")
            server.send_message(msg)
            logger.info(f"Email sent successfully to {to_email}")
        finally:
            server.quit()
    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP authentication failed: {str(e)}")
        raise HTTPException(status_code=500,
                            detail="Failed to authenticate with SMTP server. Please check SMTP credentials.")
    except smtplib.SMTPConnectError as e:
        logger.error(f"SMTP connection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to connect to SMTP server: {str(e)}")
    except TimeoutError as e:
        logger.error(f"SMTP timeout error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SMTP connection timed out: {str(e)}")
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")


# Routes
@app.get("/")
async def root():
    return {"message": "NUESA Voting System API"}


# Authentication routes
@app.post("/api/v1/auth/register", response_model=dict)
async def register(
        firstName: str = Form(...),
        lastName: str = Form(...),
        email: str = Form(...),
        matricNumber: str = Form(...),
        password: str = Form(...),
        department: str = Form(...),
        level: str = Form(...),
        profileImage: Optional[UploadFile] = File(None)
):
    existing_user = await users_collection.find_one({
        "$or": [
            {"email": email},
            {"matricNumber": matricNumber}
        ]
    })

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="User with this email or matric number already exists"
        )

    # Validate department
    department_exists = await departments_collection.find_one({"name": department})
    if not department_exists:
        raise HTTPException(status_code=400, detail="Invalid department")

    profile_image_url = None
    if profileImage:
        file_content = await profileImage.read()
        profile_image_url = upload_to_cloudinary(file_content, "profiles")

    user_data = UserCreate(
        firstName=firstName,
        lastName=lastName,
        email=email,
        matricNumber=matricNumber.upper(),
        password=password,
        department=department,
        level=level,
        profileImage=profile_image_url
    ).dict()

    hashed_password = get_password_hash(password)
    user_data["password"] = hashed_password
    user_data["is_admin"] = False
    user_data["created_at"] = datetime.utcnow()
    user_data["updated_at"] = datetime.utcnow()

    result = await users_collection.insert_one(user_data)

    return {"message": "User registered successfully", "user_id": str(result.inserted_id)}


@app.post("/api/v1/auth/login", response_model=Token)
async def login(credentials: AuthModel):
    matric_number = credentials.matricNumber
    password = credentials.password

    if not matric_number or not password:
        raise HTTPException(
            status_code=400,
            detail="Matric number and password are required"
        )

    user = await users_collection.find_one({"matricNumber": matric_number.upper()})

    if not user or not verify_password(password, user["password"]):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user["_id"])}, expires_delta=access_token_expires
    )

    user_data = {**user, "id": str(user["_id"])}
    user_response = UserResponse(**user_data)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user_response
    }


@app.post("/api/v1/auth/forgot-password", response_model=dict)
async def forgot_password(request: ForgotPasswordRequest):
    user = await users_collection.find_one({"email": request.email})
    if not user:
        logger.info(f"No user found for email {request.email}, returning success to prevent enumeration")
        return {"message": "If an account exists, a password reset link has been sent"}

    reset_token = generate_reset_token()
    expires_at = datetime.utcnow() + timedelta(hours=RESET_TOKEN_EXPIRE_HOURS)

    await password_reset_tokens_collection.insert_one({
        "user_id": str(user["_id"]),
        "token": reset_token,
        "expires_at": expires_at,
        "created_at": datetime.utcnow(),
        "used": False
    })

    reset_url = f"{BASE_URL}/reset-password/{reset_token}"
    send_reset_email(request.email, reset_url)

    return {"message": "Password reset link sent to your email"}


@app.get("/reset-password/{token}", response_class=HTMLResponse)
async def get_reset_password_form(request: Request, token: str):
    reset_token = await password_reset_tokens_collection.find_one({
        "token": token,
        "used": False,
        "expires_at": {"$gt": datetime.utcnow()}
    })

    if not reset_token:
        logger.warning(f"Invalid or expired reset token: {token}")
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Reset Password - NUESA Voting System</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 600px;
                    margin: 20px auto;
                    padding: 20px;
                    background-color: #f4f4f4;
                }
                .container {
                    background-color: #ffffff;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                h2 {
                    color: #333333;
                    text-align: center;
                }
                .error {
                    color: red;
                    text-align: center;
                    margin-bottom: 15px;
                }
                .footer {
                    text-align: center;
                    color: #999999;
                    font-size: 12px;
                    margin-top: 20px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Reset Your Password</h2>
                <p class="error">Invalid or expired reset token. Please request a new password reset link.</p>
                <div class="footer">
                    © 2025 NUESA Voting System. All rights reserved.
                </div>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=400)

    logger.info(f"Rendering reset password form for valid token: {token}")
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reset Password - NUESA Voting System</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 600px;
                margin: 20px auto;
                padding: 20px;
                background-color: #f4f4f4;
            }}
            .container {{
                background-color: #ffffff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h2 {{
                color: #333333;
                text-align: center;
            }}
            .form-group {{
                margin-bottom: 15px;
            }}
            label {{
                display: block;
                margin-bottom: 5px;
                color: #555555;
            }}
            input[type="password"] {{
                width: 100%;
                padding: 8px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                box-sizing: border-box;
            }}
            button {{
                width: 100%;
                padding: 10px;
                background-color: #4f46e5;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-weight: bold;
            }}
            button:hover {{
                background-color: #4338ca;
            }}
            .error-message {{
                color: red;
                text-align: center;
                margin-top: 10px;
                display: none;
            }}
            .success-message {{
                color: green;
                text-align: center;
                margin-top: 10px;
                display: none;
            }}
            .footer {{
                text-align: center;
                color: #999999;
                font-size: 12px;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Reset Your Password</h2>
            <form id="reset-password-form">
                <input type="hidden" id="token" value="{token}">
                <div class="form-group">
                    <label for="new_password">New Password</label>
                    <input type="password" id="new_password" name="new_password" required minlength="8">
                </div>
                <button type="submit">Reset Password</button>
            </form>
            <p id="error-message" class="error-message"></p>
            <p id="success-message" class="success-message"></p>
            <div class="footer">
                © 2025 NUESA Voting System. All rights reserved.
            </div>
        </div>
        <script>
            document.getElementById('reset-password-form').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const token = document.getElementById('token').value;
                const newPassword = document.getElementById('new_password').value;
                const errorMessage = document.getElementById('error-message');
                const successMessage = document.getElementById('success-message');

                errorMessage.style.display = 'none';
                successMessage.style.display = 'none';

                try {{
                    const response = await fetch('{BASE_URL}/api/v1/auth/reset-password', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                            'Accept': 'application/json'
                        }},
                        body: JSON.stringify({{
                            token: token,
                            new_password: newPassword
                        }})
                    }});

                    const data = await response.json();

                    if (!response.ok) {{
                        throw new Error(data.detail || 'Failed to reset password');
                    }}

                    successMessage.textContent = data.message || 'Password reset successfully';
                    successMessage.style.display = 'block';
                    document.getElementById('reset-password-form').reset();
                }} catch (error) {{
                    errorMessage.textContent = error.message;
                    errorMessage.style.display = 'block';
                }}
            }});
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/api/v1/auth/reset-password", response_model=dict)
async def reset_password(request: ResetPasswordRequest):
    if not request.new_password or len(request.new_password) < 8:
        logger.error("Password validation failed: less than 8 characters")
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long")

    reset_token = await password_reset_tokens_collection.find_one({
        "token": request.token,
        "used": False,
        "expires_at": {"$gt": datetime.utcnow()}
    })

    if not reset_token:
        logger.warning(f"Invalid or expired reset token: {request.token}")
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    await password_reset_tokens_collection.update_one(
        {"_id": reset_token["_id"]},
        {"$set": {"used": True, "updated_at": datetime.utcnow()}}
    )

    hashed_password = get_password_hash(request.new_password)
    result = await users_collection.update_one(
        {"_id": ObjectId(reset_token["user_id"])},
        {"$set": {"password": hashed_password, "updated_at": datetime.utcnow()}}
    )

    if result.matched_count == 0:
        logger.error(f"User not found for user_id: {reset_token['user_id']}")
        raise HTTPException(status_code=404, detail="User not found")

    logger.info(f"Password reset successfully for user_id: {reset_token['user_id']}")
    return {"message": "Password reset successfully"}


# User routes
@app.get("/api/v1/users/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    user_data = {**current_user, "id": str(current_user["_id"])}
    return UserResponse(**user_data)


@app.put("/api/v1/users/profile", response_model=UserResponse)
async def update_user_profile(
        firstName: Optional[str] = Form(None),
        lastName: Optional[str] = Form(None),
        email: Optional[str] = Form(None),
        level: Optional[str] = Form(None),
        profileImage: Optional[UploadFile] = File(None),
        current_user: dict = Depends(get_current_user)
):
    update_data = UserUpdate(
        firstName=firstName,
        lastName=lastName,
        email=email,
        level=level,
        profileImage=None
    ).dict(exclude_unset=True)

    if "email" in update_data:
        existing_user = await users_collection.find_one({
            "email": update_data["email"],
            "_id": {"$ne": ObjectId(current_user["_id"])}
        })
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already in use")

    if "level" in update_data:
        level_exists = await levels_collection.find_one({"name": update_data["level"]})
        if not level_exists:
            raise HTTPException(status_code=400, detail="Invalid level")

    if profileImage:
        file_content = await profileImage.read()
        profile_image_url = upload_to_cloudinary(file_content, "profiles")
        update_data["profileImage"] = profile_image_url

    if not update_data:
        raise HTTPException(status_code=400, detail="No fields provided for update")

    update_data["updated_at"] = datetime.utcnow()

    result = await users_collection.update_one(
        {"_id": ObjectId(current_user["_id"])},
        {"$set": update_data}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")

    updated_user = await users_collection.find_one({"_id": ObjectId(current_user["_id"])})
    user_data = {**updated_user, "id": str(updated_user["_id"])}
    return UserResponse(**user_data)


# Election routes
@app.get("/api/v1/elections/current", response_model=Optional[ElectionResponse])
async def get_current_election(current_user: dict = Depends(get_current_user)):
    query = {
        "status": "active",
        "$or": [
            {"election_type": "faculty"},
            {"election_type": "departmental", "department": current_user["department"]}
        ]
    }
    election = await elections_collection.find_one(query, sort=[("created_at", -1)])

    if election:
        election_data = {**election, "id": str(election["_id"])}
        return ElectionResponse(**election_data)
    return None


@app.get("/api/v1/elections", response_model=List[ElectionResponse])
async def get_elections(current_user: dict = Depends(get_current_user)):
    query = {
        "$or": [
            {"election_type": "faculty"},
            {"election_type": "departmental", "department": current_user["department"]}
        ]
    }
    elections = []
    async for election in elections_collection.find(query).sort("created_at", -1):
        election_data = {**election, "id": str(election["_id"])}
        elections.append(ElectionResponse(**election_data))
    return elections


# Admin routes for election management
@app.post("/api/v1/admin/elections", response_model=dict)
async def create_election(
        election_data: ElectionCreate,
        admin_user: dict = Depends(get_admin_user)
):
    election_dict = election_data.dict()

    election_type_exists = await election_types_collection.find_one({"key": election_dict["election_type"]})
    if not election_type_exists:
        raise HTTPException(status_code=400, detail="Invalid election type")

    if election_dict["election_type"] == "departmental":
        if not election_dict["department"]:
            raise HTTPException(status_code=400, detail="Department is required for departmental elections")
        department_exists = await departments_collection.find_one({"name": election_dict["department"]})
        if not department_exists:
            raise HTTPException(status_code=400, detail="Invalid department")
    else:
        election_dict["department"] = None

    election_dict["created_at"] = datetime.utcnow()
    election_dict["updated_at"] = datetime.utcnow()
    election_dict["created_by"] = str(admin_user["_id"])

    result = await elections_collection.insert_one(election_dict)

    return {"message": "Election created successfully", "election_id": str(result.inserted_id)}


@app.put("/api/v1/admin/elections/{election_id}", response_model=dict)
async def update_election(
        election_id: str,
        election_data: ElectionUpdate,
        admin_user: dict = Depends(get_admin_user)
):
    update_dict = {k: v for k, v in election_data.dict().items() if v is not None}
    if not update_dict:
        raise HTTPException(status_code=400, detail="No fields provided for update")

    if "election_type" in update_dict:
        election_type_exists = await election_types_collection.find_one({"key": update_dict["election_type"]})
        if not election_type_exists:
            raise HTTPException(status_code=400, detail="Invalid election type")

        if update_dict["election_type"] == "departmental":
            if "department" not in update_dict or not update_dict["department"]:
                raise HTTPException(status_code=400, detail="Department is required for departmental elections")
            department_exists = await departments_collection.find_one({"name": update_dict["department"]})
            if not department_exists:
                raise HTTPException(status_code=400, detail="Invalid department")
        else:
            update_dict["department"] = None

    update_dict["updated_at"] = datetime.utcnow()

    result = await elections_collection.update_one(
        {"_id": ObjectId(election_id)},
        {"$set": update_dict}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Election not found")

    return {"message": "Election updated successfully"}


@app.delete("/api/v1/admin/elections/{election_id}", response_model=dict)
async def delete_election(
        election_id: str,
        admin_user: dict = Depends(get_admin_user)
):
    result = await elections_collection.delete_one({"_id": ObjectId(election_id)})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Election not found")

    return {"message": "Election deleted successfully"}


# Election Type and Position Management Routes
@app.get("/api/v1/admin/election-types", response_model=List[ElectionTypeResponse])
async def get_admin_election_types(admin_user: dict = Depends(get_admin_user)):
    election_types = []
    async for election_type in election_types_collection.find():
        election_type_data = {**election_type, "id": str(election_type["_id"])}
        election_types.append(ElectionTypeResponse(**election_type_data))
    return election_types


@app.post("/api/v1/admin/election-types", response_model=dict)
async def create_election_type(
        election_type_data: ElectionTypeCreate,
        admin_user: dict = Depends(get_admin_user)
):
    election_type_dict = election_type_data.dict()
    election_type_dict["created_at"] = datetime.utcnow()
    election_type_dict["updated_at"] = datetime.utcnow()
    election_type_dict["created_by"] = str(admin_user["_id"])

    result = await election_types_collection.insert_one(election_type_dict)

    return {"message": "Election type created successfully", "election_type_id": str(result.inserted_id)}


@app.put("/api/v1/admin/election-types/{election_type_id}", response_model=dict)
async def update_election_type(
        election_type_id: str,
        election_type_data: ElectionTypeUpdate,
        admin_user: dict = Depends(get_admin_user)
):
    update_dict = {k: v for k, v in election_type_data.dict().items() if v is not None}
    if not update_dict:
        raise HTTPException(status_code=400, detail="No fields provided for update")

    update_dict["updated_at"] = datetime.utcnow()

    result = await election_types_collection.update_one(
        {"_id": ObjectId(election_type_id)},
        {"$set": update_dict}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Election type not found")

    return {"message": "Election type updated successfully"}


@app.delete("/api/v1/admin/election-types/{election_type_id}", response_model=dict)
async def delete_election_type(
        election_type_id: str,
        admin_user: dict = Depends(get_admin_user)
):
    result = await election_types_collection.delete_one({"_id": ObjectId(election_type_id)})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Election type not found")

    return {"message": "Election type deleted successfully"}


@app.get("/api/v1/admin/positions", response_model=List[PositionResponse])
async def get_admin_positions(
        election_type: Optional[str] = None,
        admin_user: dict = Depends(get_admin_user)
):
    query = {}
    if election_type:
        query["election_type"] = election_type

    positions = []
    async for position in positions_collection.find(query):
        position_data = {**position, "id": str(position["_id"])}
        positions.append(PositionResponse(**position_data))
    return positions


@app.post("/api/v1/admin/positions", response_model=dict)
async def create_position(
        position_data: PositionCreate,
        admin_user: dict = Depends(get_admin_user)
):
    position_dict = position_data.dict()
    position_dict["created_at"] = datetime.utcnow()
    position_dict["updated_at"] = datetime.utcnow()
    position_dict["created_by"] = str(admin_user["_id"])

    result = await positions_collection.insert_one(position_dict)

    return {"message": "Position created successfully", "position_id": str(result.inserted_id)}


@app.put("/api/v1/admin/positions/{position_id}", response_model=dict)
async def update_position(
        position_id: str,
        position_data: PositionUpdate,
        admin_user: dict = Depends(get_admin_user)
):
    update_dict = {k: v for k, v in position_data.dict().items() if v is not None}
    if not update_dict:
        raise HTTPException(status_code=400, detail="No fields provided for update")

    update_dict["updated_at"] = datetime.utcnow()

    result = await positions_collection.update_one(
        {"_id": ObjectId(position_id)},
        {"$set": update_dict}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Position not found")

    return {"message": "Position updated successfully"}


@app.delete("/api/v1/admin/positions/{position_id}", response_model=dict)
async def delete_position(
        position_id: str,
        admin_user: dict = Depends(get_admin_user)
):
    result = await positions_collection.delete_one({"_id": ObjectId(position_id)})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Position not found")

    return {"message": "Position deleted successfully"}


# Candidate routes
@app.get("/api/v1/candidates", response_model=List[CandidateResponse])
async def get_candidates(
        election_id: Optional[str] = None,
        election_type: Optional[str] = None,
        department: Optional[str] = None,
        position: Optional[str] = None,
        current_user: dict = Depends(get_current_user)
):
    query = {}
    if election_id:
        query["election_id"] = election_id
    if election_type:
        if election_type == "departmental":
            query["department"] = current_user["department"]
        query["election_type"] = election_type
    if department:
        query["department"] = department
    if position:
        query["position"] = position

    election_ids = []
    async for election in elections_collection.find({
        "$or": [
            {"election_type": "faculty"},
            {"election_type": "departmental", "department": current_user["department"]}
        ]
    }):
        election_ids.append(str(election["_id"]))
    query["election_id"] = {"$in": election_ids}

    candidates = []
    async for candidate in candidates_collection.find(query).sort("position", 1):
        candidate_data = {**candidate, "id": str(candidate["_id"])}
        candidates.append(CandidateResponse(**candidate_data))
    return candidates


@app.get("/api/v1/positions", response_model=dict)
async def get_positions(
        election_id: Optional[str] = None,
        election_type: Optional[str] = None,
        current_user: dict = Depends(get_current_user)
):
    query = {}
    if election_id:
        query["election_id"] = election_id
    if election_type:
        query["election_type"] = election_type
        if election_type == "departmental":
            query["department"] = current_user["department"]

    election_ids = []
    async for election in elections_collection.find({
        "$or": [
            {"election_type": "faculty"},
            {"election_type": "departmental", "department": current_user["department"]}
        ]
    }):
        election_ids.append(str(election["_id"]))
    query["election_id"] = {"$in": election_ids}

    positions = await candidates_collection.distinct("position", query)
    return {"positions": positions}


@app.get("/api/v1/elections/types", response_model=dict)
async def get_election_types():
    election_types = {}
    async for election_type in election_types_collection.find():
        type_key = election_type["key"]
        election_types[type_key] = {
            "name": election_type["name"],
            "description": election_type["description"],
            "positions": []
        }

    async for position in positions_collection.find():
        election_type = position["election_type"]
        if election_type in election_types:
            election_types[election_type]["positions"].append(position["name"])

    if not election_types:
        election_types = {
            "faculty": {
                "name": "Faculty Elections",
                "description": "Elections for NUESA Faculty positions - All students can vote",
                "positions": [
                    "President", "Vice President", "General Secretary",
                    "Assistant General Secretary", "Treasurer", "Financial Secretary",
                    "Public Relations Officer (PRO)", "Social Director",
                    "Sports Director", "Welfare Director"
                ]
            },
            "departmental": {
                "name": "Departmental Elections",
                "description": "Elections for Department-specific positions - Only department students can vote",
                "positions": [
                    "President", "Vice President", "Secretary",
                    "Assistant Secretary", "Treasurer", "PRO",
                    "Social Director", "Sports Director"
                ]
            }
        }

    return election_types


@app.post("/api/v1/admin/candidates", response_model=dict)
async def create_candidate(
        fullName: str = Form(...),
        position: str = Form(...),
        election_id: str = Form(...),
        election_type: str = Form(...),
        department: Optional[str] = Form(None),
        level: str = Form(...),
        manifesto: str = Form(...),
        photo: Optional[UploadFile] = File(None),
        admin_user: dict = Depends(get_admin_user)
):
    election = await elections_collection.find_one({"_id": ObjectId(election_id)})
    if not election:
        raise HTTPException(status_code=404, detail="Election not found")

    if election["election_type"] != election_type:
        raise HTTPException(status_code=400, detail="Election type mismatch")

    if election_type == "departmental":
        if not department or department != election["department"]:
            raise HTTPException(status_code=400, detail="Department must match the election's department")
        department_exists = await departments_collection.find_one({"name": department})
        if not department_exists:
            raise HTTPException(status_code=400, detail="Invalid department")
    else:
        if department:
            raise HTTPException(status_code=400, detail="Department should not be provided for faculty elections")

    photo_url = None
    if photo:
        file_content = await photo.read()
        photo_url = upload_to_cloudinary(file_content, "candidates")

    candidate_data = CandidateCreate(
        fullName=fullName,
        position=position,
        election_id=election_id,
        election_type=election_type,
        department=department,
        level=level,
        manifesto=manifesto,
        photo=photo_url
    ).dict()

    candidate_data["vote_count"] = 0
    candidate_data["created_at"] = datetime.utcnow()
    candidate_data["updated_at"] = datetime.utcnow()
    candidate_data["created_by"] = str(admin_user["_id"])

    result = await candidates_collection.insert_one(candidate_data)

    return {"message": "Candidate created successfully", "candidate_id": str(result.inserted_id)}


# Voting routes
@app.post("/api/v1/votes", response_model=dict)
async def cast_vote(
        vote_data: VoteCreate,
        current_user: dict = Depends(get_current_user)
):
    candidate_id = vote_data.candidate_id

    candidate = await candidates_collection.find_one({"_id": ObjectId(candidate_id)})
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    election = await elections_collection.find_one({"_id": ObjectId(candidate["election_id"])})
    if not election:
        raise HTTPException(status_code=404, detail="Election not found")

    if election["status"] != "active":
        raise HTTPException(status_code=400, detail="Election is not active")

    if vote_data.election_type != election["election_type"]:
        raise HTTPException(status_code=400, detail="Election type mismatch")
    if vote_data.election_id != str(election["_id"]):
        raise HTTPException(status_code=400, detail="Election ID mismatch")
    if vote_data.position != candidate["position"]:
        raise HTTPException(status_code=400, detail="Position mismatch")
    if vote_data.department != candidate["department"]:
        raise HTTPException(status_code=400, detail="Department mismatch")

    if election["election_type"] == "departmental":
        if current_user["department"] != election["department"]:
            raise HTTPException(
                status_code=403,
                detail="You can only vote for candidates in your department"
            )

    existing_vote = await votes_collection.find_one({
        "user_id": str(current_user["_id"]),
        "position": candidate["position"],
        "election_id": candidate["election_id"]
    })

    if existing_vote:
        raise HTTPException(status_code=400, detail=f"Already voted for {candidate['position']} in this election")

    vote_record = vote_data.dict()
    vote_record["user_id"] = str(current_user["_id"])
    vote_record["created_at"] = datetime.utcnow()

    await votes_collection.insert_one(vote_record)

    await candidates_collection.update_one(
        {"_id": ObjectId(candidate_id)},
        {"$inc": {"vote_count": 1}}
    )

    return {"message": "Vote cast successfully"}


@app.post("/api/v1/votes/batch", response_model=BatchVoteResponse)
async def cast_batch_votes(
        vote_data: BatchVoteCreate,
        current_user: dict = Depends(get_current_user)
):
    votes = vote_data.votes
    if not votes:
        raise HTTPException(status_code=400, detail="No votes provided")

    successful_votes = []
    failed_votes = []

    for vote in votes:
        try:
            candidate_id = vote.candidate_id
            candidate = await candidates_collection.find_one({"_id": ObjectId(candidate_id)})
            if not candidate:
                failed_votes.append({"candidate_id": candidate_id, "error": "Candidate not found"})
                continue

            election = await elections_collection.find_one({"_id": ObjectId(candidate["election_id"])})
            if not election:
                failed_votes.append({"candidate_id": candidate_id, "error": "Election not found"})
                continue

            if election["status"] != "active":
                failed_votes.append({"candidate_id": candidate_id, "error": "Election is not active"})
                continue

            if vote.election_type != election["election_type"]:
                failed_votes.append({"candidate_id": candidate_id, "error": "Election type mismatch"})
                continue
            if vote.election_id != str(election["_id"]):
                failed_votes.append({"candidate_id": candidate_id, "error": "Election ID mismatch"})
                continue
            if vote.position != candidate["position"]:
                failed_votes.append({"candidate_id": candidate_id, "error": "Position mismatch"})
                continue
            if vote.department != candidate["department"]:
                failed_votes.append({"candidate_id": candidate_id, "error": "Department mismatch"})
                continue

            if election["election_type"] == "departmental":
                if current_user["department"] != election["department"]:
                    failed_votes.append({
                        "candidate_id": candidate_id,
                        "error": "Can only vote for candidates in your department"
                    })
                    continue

            existing_vote = await votes_collection.find_one({
                "user_id": str(current_user["_id"]),
                "position": candidate["position"],
                "election_id": candidate["election_id"]
            })

            if existing_vote:
                failed_votes.append({
                    "candidate_id": candidate_id,
                    "error": f"Already voted for {candidate['position']} in this election"
                })
                continue

            vote_record = vote.dict()
            vote_record["user_id"] = str(current_user["_id"])
            vote_record["created_at"] = datetime.utcnow()

            await votes_collection.insert_one(vote_record)

            await candidates_collection.update_one(
                {"_id": ObjectId(candidate_id)},
                {"$inc": {"vote_count": 1}}
            )

            successful_votes.append({
                "candidate_id": candidate_id,
                "position": candidate["position"],
                "election_id": vote.election_id,
                "election_type": vote.election_type,
                "department": vote.department
            })

        except ValueError as e:
            failed_votes.append({
                "candidate_id": vote.candidate_id,
                "error": str(e)
            })
        except Exception as e:
            failed_votes.append({
                "candidate_id": vote.candidate_id,
                "error": f"Unexpected error: {str(e)}"
            })

    response = BatchVoteResponse(
        message=f"Processed {len(votes)} votes",
        successful=len(successful_votes),
        failed=len(failed_votes),
        successful_votes=successful_votes,
        failed_votes=failed_votes
    )

    if response.failed > 0 and response.successful == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=response.dict()
        )

    return response


@app.get("/api/v1/votes/status", response_model=VotingStatusResponse)
async def get_voting_status(
        election_id: Optional[str] = None,
        election_type: Optional[str] = None,
        current_user: dict = Depends(get_current_user)
):
    query = {"user_id": str(current_user["_id"])}
    if election_id:
        query["election_id"] = election_id
    if election_type:
        query["election_type"] = election_type

    voted_positions = []
    async for vote in votes_collection.find(query):
        voted_positions.append(vote["position"])

    return VotingStatusResponse(
        voted_positions=voted_positions,
        total_votes=len(voted_positions)
    )


# Results routes
@app.get("/api/v1/results", response_model=List[CandidateResponse])
async def get_results(
        election_id: Optional[str] = None,
        election_type: Optional[str] = None,
        department: Optional[str] = None,
        current_user: dict = Depends(get_current_user)
):
    query = {}
    if election_id:
        query["election_id"] = election_id
    if election_type:
        query["election_type"] = election_type
    if department:
        query["department"] = department
    if election_type == "departmental":
        query["department"] = current_user["department"]

    election_ids = []
    async for election in elections_collection.find({
        "$or": [
            {"election_type": "faculty"},
            {"election_type": "departmental", "department": current_user["department"]}
        ]
    }):
        election_ids.append(str(election["_id"]))
    query["election_id"] = {"$in": election_ids}

    results = []
    async for candidate in candidates_collection.find(query).sort("vote_count", -1):
        candidate_data = {**candidate, "id": str(candidate["_id"])}
        results.append(CandidateResponse(**candidate_data))
    return results


# Announcements routes
@app.get("/api/v1/announcements", response_model=List[AnnouncementResponse])
async def get_announcements():
    announcements = []
    async for announcement in announcements_collection.find().sort("created_at", -1).limit(10):
        announcement_data = {**announcement, "id": str(announcement["_id"])}
        announcements.append(AnnouncementResponse(**announcement_data))
    return announcements


@app.post("/api/v1/admin/announcements", response_model=dict)
async def create_announcement(
        announcement_data: AnnouncementCreate,
        admin_user: dict = Depends(get_admin_user)
):
    announcement_dict = announcement_data.dict()
    announcement_dict["created_at"] = datetime.utcnow()
    announcement_dict["created_by"] = str(admin_user["_id"])

    result = await announcements_collection.insert_one(announcement_dict)

    return {"message": "Announcement created successfully", "announcement_id": str(result.inserted_id)}


# Department management routes
@app.get("/api/v1/departments", response_model=List[DepartmentResponse])
async def get_departments():
    departments = []
    async for dept in departments_collection.find():
        dept_data = {**dept, "id": str(dept["_id"])}
        departments.append(DepartmentResponse(**dept_data))
    return departments


@app.post("/api/v1/admin/departments", response_model=dict)
async def create_department(
        department_data: DepartmentCreate,
        admin_user: dict = Depends(get_admin_user)
):
    department_dict = department_data.dict()
    department_dict["created_at"] = datetime.utcnow()

    result = await departments_collection.insert_one(department_dict)

    return {"message": "Department created successfully", "department_id": str(result.inserted_id)}


# Level management routes
@app.get("/api/v1/levels", response_model=List[LevelResponse])
async def get_levels():
    levels = []
    async for level in levels_collection.find():
        level_data = {**level, "id": str(level["_id"])}
        levels.append(LevelResponse(**level_data))
    return levels


@app.post("/api/v1/admin/levels", response_model=dict)
async def create_level(
        level_data: LevelCreate,
        admin_user: dict = Depends(get_admin_user)
):
    level_dict = level_data.dict()
    level_dict["created_at"] = datetime.utcnow()

    result = await levels_collection.insert_one(level_dict)

    return {"message": "Level created successfully", "level_id": str(result.inserted_id)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)