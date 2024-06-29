from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
import openai

import re
import math

# Load environment variables from .env file


# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Global variable to store the uploaded data
global_data = None


# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/second')
def second_page():
    return render_template('second.html')

# Route for file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    global global_data
    global global_datapd

    if 'document' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['document']

    if file and file.filename.endswith('.csv'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            global_data = pd.read_csv(file_path)
            return jsonify({"message": "File uploaded successfully", "columns": global_data.columns.tolist()}), 201
        except Exception as e:
            return jsonify({"error": f"Error reading CSV file: {str(e)}"}), 400
    else:
        return jsonify({"error": "Invalid file format. Please upload a CSV file."}), 400

# Function to generate and verify query
def generate_and_verify_query(user_input, global_data, max_attempts=1):
    
    column_names = ", ".join([f"'{col}'" for col in global_data.columns])
    for attempt in range(max_attempts):
        # Generate query using OpenAI
        ai_response = generate_query_with_openai(user_input, column_names)

        # Extract the pandas query from the AI response
        pandas_query = extract_pandas_query(ai_response)
        print("pandas", pandas_query)

        return pandas_query

# Function to generate query using OpenAI
def generate_query_with_openai(user_input, column_names):
    SYSTEM_PROMPT = f"""
    # You are an AI assistant analyzing CSV data stored in a pandas DataFrame named 'global_data'.
    # Available columns: {column_names}.

    # CRITICAL RULES:
    # 1. Return a SINGLE pandas query starting with 'global_data.'.
    # 2. Use 'global_data.assign()' for new columns.
    # 3. ALWAYS use EXACT column names as provided in the available columns list.
    # 4. Enclose column names in square brackets and double quotes, e.g. global_data["Exact Column Name"].
    # 5. DO NOT use any float() or other type conversions in the query.
    # 6. Include '.sort_values()' for ranking or sorting.
    # 7. End queries with column selection using double brackets.
    # 8. Use explicit numerical values: 0.5, 0.25, etc.
    # 9. Use pd.to_numeric(global_data["Column Name"], errors='coerce') for numeric conversions.

    # Before writing the query, list the exact column names you will use for:
    # 1. Patient Incidence
    # 2. Recruitment Rate
    # 3. Percentage of sites with no competitor trials
    # 4. Country

    # Then provide the query using these exact column names.

    # If the requested data is not present, respond with: "DATA_NOT_PRESENT: <explanation>"

    # Format your response as:
    # COLUMN NAMES:
    # 1. Patient Incidence: "Exact Column Name"
    # 2. Recruitment Rate: "Exact Column Name"
    # 3. Percentage of sites: "Exact Column Name"
    # 4. Country: "Exact Column Name"

    # PANDAS_QUERY: <single_line_query>

    # User Input: {user_input}
    Please note that the global variable name is global_data. global_datapd or anything are wrong and must not be used
    Please read the error and correct the query. The repeat query should not be same as before
    If user inputs that there is something wrong with query, you must upadte so that it runs
    It will have the error and query, you need to fix that accordingly

    If user does not ask anything, say Hi, hello or anything, respond Hi, how may I help you?
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(column_names=column_names)},
        {"role": "user", "content": user_input}
    ]
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        temperature=0,
        max_tokens=450
    )
    return response.choices[0].message.content

# Function to extract pandas query from AI response
def extract_pandas_query(ai_response):
    match = re.search(r'PANDAS_QUERY:\s*(.*)', ai_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def process_query(query):
    # Replace column references with pd.to_numeric to ensure proper conversion
    query = re.sub(r'global_data\["([^"]+)"\]', r'pd.to_numeric(global_data["\1"], errors="coerce")', query)

    # Replace numeric constants with pd.to_numeric to ensure consistency in operations
    query = re.sub(r'([-+]?\d*\.\d+|\d+)', r'pd.to_numeric(\1, errors="coerce")', query)

    return query

def format_result(result):
    if isinstance(result, dict) and "Result" in result:
        return result["Result"]  # Return the value associated with "Result" key

    elif isinstance(result, pd.Series):
        # Check if the result is a Series (e.g., country scores)
        formatted_result = []
        for country, score in result.items():
            if pd.notna(score):  # Exclude NaN values
                formatted_result.append({"Country": country, "Score": score})
        return formatted_result

    elif isinstance(result, (int, float, str)):
        return result  # Return directly if it's a single value (e.g., country name)

    else:
        return "Unsupported result format"
    

# Function to verify and execute the query
def verify_and_execute_query(query):
    global global_data
    global_datapd = global_data.copy()  # Create a copy for processing
    
    
    if bool(re.search(r'\d+(\.\d+)?', query)):
        string_expression = process_query(query)
    else:
        string_expression = query

    result = eval(string_expression, {"global_data": global_data, "pd": pd})
    return result
    

# Route for chat
@app.route('/chat', methods=['POST'])
def chat():
    global global_data

    if global_data is None:
        return jsonify({"error": "No data uploaded. Please upload a CSV file first."}), 400

    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    result = generate_and_verify_query(user_input, global_data, max_attempts=1)
    result1 = verify_and_execute_query(result)
    print(result1)
    
    if isinstance(result1, pd.DataFrame) or isinstance(result1, pd.Series):
        result1 = result1.dropna(axis=1).to_dict(orient='records')
        first_item = result1[0]  # Get the first dictionary in the list
        first_key = next(iter(first_item))
        values_list = [item[first_key] for item in result1]
        return jsonify({"Result": values_list}), 200
    elif isinstance(result1, (int, float, str)):
            return jsonify(format_result(result1))
    else:
            return jsonify(result1)


if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, render_template, jsonify, send_file
import os
import requests
import pandas as pd
from io import BytesIO
import openai
import re
import json

import pymongo



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

API_KEY = os.getenv('MONGODB_API_KEY')
DATABASE_NAME = 'Upwork_Chirag'
BASE_URL = 'https://eu-west-2.aws.data.mongodb-api.com/app/data-adgmytr/endpoint/data/v1/action'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

openai.api_key = os.getenv('OPENAI_API_KEY')

@app.route('/')
def index():
    return render_template('index.html')  # Render index.html from templates folder

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'document' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['document']

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        collection_name = 'UP_Chirag'  # Change this line

        headers = {
            'Content-Type': 'application/json',
            'Access-Control-Request-Headers': '*',
            'api-key': API_KEY,
        }

        with open(file_path, 'rb') as f:
            file_data = f.read()

        payload = json.dumps({
            "collection": collection_name,
            "database": DATABASE_NAME,
            "dataSource": "Cluster0",
            "document": {
                "filename": file.filename,
                "path": file_path,
                "data": file_data.decode('utf-8')
            }
        })

        response = requests.post(f"{BASE_URL}/insertOne", headers=headers, data=payload)

        if response.status_code == 201:
            return jsonify({"message": "File uploaded and collection created successfully"}), 201
        else:
            return jsonify({"error": f"An error occurred: {response.text}"}), 500

    return jsonify({"error": "Unknown error occurred"}), 500

@app.route('/get_tables', methods=['GET'])
def get_tables():
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Request-Headers': '*',
        'api-key': API_KEY,
    }
    payload = json.dumps({
        "dataSource": "Cluster0",
        "database": DATABASE_NAME,
        "collection": "UP_Chirag",
        "pipeline": [
            {"$project": {"filename": 1, "_id": 0}}
        ]
    })
    try:
        response = requests.post(f"{BASE_URL}/aggregate", headers=headers, data=payload)
        response.raise_for_status()  # Raise an exception for non-2xx status codes
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

    if response.status_code == 200:
        documents = response.json().get('documents', [])
        table_names = [doc['filename'] for doc in documents]
        return jsonify({"tables": table_names})
    else:
        error_message = response.text
        return jsonify({"error": f"An error occurred: {error_message}"}), 500


@app.route('/create_ranking', methods=['POST'])
def create_ranking():
    data = request.json
    table_name = data['table']
    patient_incidence_weight = data['patientIncidence'] / 100.0
    competition_weight = data['competition'] / 100.0
    recruitment_rate_weight = data['recruitmentRate'] / 100.0

    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Request-Headers': '*',
        'api-key': API_KEY,
    }

    payload = json.dumps({
        "dataSource": "Cluster0",
        "database": DATABASE_NAME,
        "collection": table_name,
        "pipeline": []
    })

    try:
        response = requests.post(f"{BASE_URL}/aggregate", headers=headers, data=payload)
        response.raise_for_status()  # Raise an exception for non-2xx status codes
        documents = response.json().get('documents', [])
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

    # Create DataFrame from retrieved documents
    df = pd.DataFrame(documents)

    # Convert relevant columns to numeric
    numeric_columns = ['COPD 2021 Incidence', '% Sites with Ongoing Competitor Trials', 'Roche Median Country Recruitment Rate']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Calculate weighted scores
    df['Patient Incidence Weighted'] = df['COPD 2021 Incidence'] * patient_incidence_weight
    df['Competition Weighted'] = (1 - df['% Sites with Ongoing Competitor Trials']) * competition_weight
    df['Recruitment Rate Weighted'] = df['Roche Median Country Recruitment Rate'] * recruitment_rate_weight

    # Select specific columns for output
    columns_to_export = ['Country', 'Patient Incidence Weighted', 'Competition Weighted', 'Recruitment Rate Weighted']
    df_export = df[columns_to_export]

    # Prepare MongoDB aggregation pipeline
    aggregation_pipeline = [
        {
            "$project": {
                "Country": 1,
                "Patient Incidence Weighted": 1,
                "Competition Weighted": 1,
                "Recruitment Rate Weighted": 1,
                "_id": 0  # Exclude the _id field from the output
            }
        }
    ]

    # Execute the aggregation pipeline on MongoDB
    try:
        aggregated_data = list(db[table_name].aggregate(aggregation_pipeline))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Create a new DataFrame from the aggregated data (if needed)
    df_aggregated = pd.DataFrame(aggregated_data)

    # Prepare Excel file
    output = BytesIO()
    df_export.to_excel(output, index=False)
    output.seek(0)

    # Return the Excel file as a downloadable attachment
    return send_file(output, attachment_filename='ranking_result.xlsx', as_attachment=True)




mongo_client = pymongo.MongoClient("mongodb+srv://kanchang12:b7wglgwWDZyxeIOW@cluster0.sle630c.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = mongo_client[DATABASE_NAME]

def fetch_metadata():
    table_names = db.list_collection_names()
    column_names = set()

    # Fetch unique column names across all tables
    for table_name in table_names:
        collection = db[table_name]
        for document in collection.find().limit(1):  # Limit to 1 document to get column names
            column_names.update(document.keys())

    return table_names, list(column_names)

# Function to fetch table schema
def fetch_table_schema(table_name):
    collection = db[table_name]
    schema = {}

    # Fetch one document from the collection to determine schema
    sample_document = collection.find_one()

    if sample_document:
        for key, value in sample_document.items():
            # Determine the type of each field based on the sample document
            field_type = determine_field_type(value)
            schema[key] = field_type

    return schema

# Function to determine the type of a field
def determine_field_type(value):
    # Map Python types to MongoDB BSON types
    if isinstance(value, str):
        return "string"
    elif isinstance(value, int):
        return "integer"
    elif isinstance(value, float):
        return "float"
    elif isinstance(value, bool):
        return "boolean"
    elif isinstance(value, list):
        return "array"
    elif isinstance(value, dict):
        return "object"
    elif isinstance(value, type(None)):
        return "null"
    else:
        return "unknown"

def update_query(query):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
{"role": "system", "content": (f"""
read the user input and format it to perfect executable mongodb query. It must be perfect and should not have any other value
remove any javascript or any other unnecessary word
   
                               """)
},
            {"role": "user", "content": query}
        ],
        temperature=0,
        max_tokens=450,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

def extract_query_body(query_str):
    # Strip leading and trailing whitespace
    query_str = query_str.strip()

    # Check if the query starts with a valid MongoDB operation like aggregate, find, etc.
    if query_str.startswith("db."):

        # Split the query string by dots to extract parts
        parts = query_str.split('.')

        # Check if there are at least three parts (db.collection.operation)
        if len(parts) >= 3:
            # Extract the third part and concatenate the rest with dots
            remaining_query = parts[2:]
            print(remaining_query)
            return remaining_query
        else:
            # If there are less than three parts, return the original query string
            return query_str

    else:
        # If the query doesn't start with 'db.', return the original query string
        return query_str


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message']

    table_names, column_names = fetch_metadata()
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
{"role": "system", "content": (f"""
# You are working as a DB Admin for MongoDB Atlas. Your job is to write queries that fetch correct records.
# Table names are available at {table_names} and column names at {column_names}.
# When the user provides input, first check these values to find the complete and correct column or table name. The user may only mention part of the name.
# To retrieve information from the MongoDB database, write Python-compatible MongoDB query code that:
#If user asks Total Patients, you need to return COPD 2021 Incidence, similarly when country number is asked, you need to return country count
# 1. Connects to MongoDB Atlas using the MONGODB_URI environment variable.
# 2. Accesses the MongoDB database Upwork_Chirag and specifies the collection (e.g., UP_Chirag).
You need to verify and find the correct column name, you can't keep using what user wrote as that would be wrong
There is nothing called USA, total_patients, and other recho factor or any other term user may say. You have to find the right column name
# 3. Constructs a valid MongoDB query based on the user's input, handling various conditions and fields.
# 4. Uses the aggregate() method for counting or aggregating data, not distinct() or count() methods.
#
# Format your response as follows: DATABASE_QUERY : <actual query> : END OF QUERY
# The query must be valid Python code that can directly interact with a MongoDB database.)
""")
},
            {"role": "user", "content": user_input}
        ],
        temperature=0,
        max_tokens=450,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    assistant_response = response.choices[0].message.content

    # Using regex to identify the query in the assistant's response
    match = re.search(r'DATABASE_QUERY\s*:\s*(.*?)\s*:\s*END\s*OF\s*QUERY', assistant_response, re.DOTALL)
    if match:
        query = match.group(1)
        
    else:
        # Handle case where no valid query is found
        return jsonify({"error": "No valid query found in the assistant's response"}), 400

    query1 = update_query(query)
    MONGODB_URI = "mongodb+srv://username:password@cluster0.sle630c.mongodb.net/Upwork_Chirag?retryWrites=true&w=majority"
    DATABASE_NAME = "Upwork_Chirag"
    collection_name = "UP_Chirag"
        
    # Connect to MongoDB
    mongo_client = pymongo.MongoClient(MONGODB_URI)
    db = mongo_client[DATABASE_NAME]
    collection = db[collection_name]
        
    # Update and extract query body
    query = update_query(query)
    query_body = extract_query_body(query)
        
    if query_body:
        # Execute the aggregation pipeline
        result = list(collection.aggregate(query_body))
        return result
    else:
        return {"error": "Invalid aggregation pipeline format"}
    

    return result


if __name__ == '__main__':
    app.run(debug=True)
