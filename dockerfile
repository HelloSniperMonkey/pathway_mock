FROM pathwaycom/pathway:latest

# Set working directory
WORKDIR /app

# Install system dependencies for OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs kyc_data user_profiles streaming_data

# Expose port for potential API usage
EXPOSE 8000

# Default: Run Pathway demo (Task 1 requirement)
# Uncomment the line below to run main Financial AI Assistant instead
CMD [ "python", "./pathway_demo.py" ]
# CMD [ "python", "./main.py" ]
