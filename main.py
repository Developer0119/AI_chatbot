from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DECIMAL
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from fastapi.middleware.cors import CORSMiddleware
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Database configuration
DATABASE_URL = "mysql+pymysql://root:root@localhost:3306/ai_chatbot_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Define database models
class Supplier(Base):
    __tablename__ = "suppliers"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    contact_info = Column(Text)
    product_categories = Column(Text)

class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    brand = Column(String)
    price = Column(DECIMAL(10, 2))
    category = Column(String)
    description = Column(Text)
    supplier_id = Column(Integer, ForeignKey("suppliers.id"))

Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI()

# Load language model (replace with an appropriate open-source LLM if needed)
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Use an available model (e.g., gpt-2)
model = AutoModelForCausalLM.from_pretrained("gpt2")  # Use an available model (e.g., gpt-2)
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Initialize summarizer
summarizer = pipeline("summarization")

# CORS configuration
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Query request model
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def chatbot_query(request: QueryRequest, db: Session = Depends(get_db)):
    query_text = request.query.lower()
    return process_query(db, query_text)

def process_query(db: Session, query: str):
    """Process user query, fetch data, and enhance response."""
    if "product by brand" in query:
        brand = query.split("brand")[-1].strip()  # Extract brand name
        products = get_products_by_brand(db, brand)
        product_list = [{"name": p.name, "price": str(p.price)} for p in products]
        prompt = f"Here are the products from brand {brand}: {json.dumps(product_list)}. Summarize this in a user-friendly and informative manner."

    elif "suppliers" in query:
        category = query.split("provide")[-1].strip()  # Extract category
        suppliers = get_suppliers_by_category(db, category)
        supplier_list = [{"name": s.name, "contact": s.contact_info} for s in suppliers]
        prompt = f"These suppliers provide products in the category '{category}': {json.dumps(supplier_list)}. Summarize this in a concise manner."

    elif "details of product" in query:
        product_name = query.split("product")[-1].strip()  # Extract product name
        product = get_product_details(db, product_name)
        if product:
            details = f"Name: {product.name}, Brand: {product.brand}, Price: {product.price}, Description: {product.description}"
            prompt = f"Here are the details of the product '{product_name}': {details}. Generate a structured summary."
        else:
            return {"message": f"Product '{product_name}' not found."}
    
    else:
        return {"message": "Query not understood."}

    # Get enhanced response from LLM
    llm_response = llm(prompt)
    return {"query": query, "response": llm_response[0]['generated_text']}

def get_products_by_brand(db: Session, brand: str):
    return db.query(Product).filter(Product.brand == brand).all()

def get_suppliers_by_category(db: Session, category: str):
    return db.query(Supplier).filter(Supplier.product_categories.like(f"%{category}%")).all()

def get_product_details(db: Session, product_name: str):
    return db.query(Product).filter(Product.name == product_name).first()

import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host='127.0.0.1', port=8000, workers=4, debug=True)
