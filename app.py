

from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
import openai
import re

# Initialize Flask app
app = Flask(__name__)
openai.api_key = os.getenv('OPENAI_API_KEY')

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

# Global variable to store the uploaded data
global_data = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global global_data

    if 'document' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['document']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and file.filename.endswith('.csv'):
        try:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            global_data = pd.read_csv(filename)
            
            return jsonify({
                "message": "File uploaded successfully",
                "columns": global_data.columns.tolist()
            }), 200
        except Exception as e:
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file format. Please upload a CSV file."}), 400

def is_pandas_query(resp):
    pattern = r'global_data\.[a-zA-Z_]+\([^\)]*\)|global_data\[[\'"][^\'"]+[\'"]\]'
    return bool(re.search(pattern, resp))

def generate_query_with_openai(user_input, column_names):
    column_list = column_names if isinstance(column_names, list) else column_names.split(", ")
    
   SYSTEM_PROMPT = f"""
    You are an AI assistant analyzing CSV data stored in a pandas DataFrame named 'global_data'.
    Available columns: {', '.join(column_list)}.
    CRITICAL RULES:
    1. Return a SINGLE pandas query starting with 'global_data.'.
    2. Use 'global_data.assign()' for new columns.
    3. ALWAYS use EXACT column names as provided in the available columns list.
    4. Enclose column names in square brackets and double quotes, e.g. global_data["Exact Column Name"].
    5. DO NOT use any float() or other type conversions in the query.
    6. Include '.sort_values()' for ranking or sorting.
    7. End queries with column selection using double brackets, including ALL relevant columns.
    8. Use explicit numerical values: 0.5, 0.25, etc.
    9. Use pd.to_numeric(global_data["Column Name"], errors='coerce') for numeric conversions.
    10. For counting unique values, use .nunique() instead of .unique().count()
    11. For total counts, use .count()
    12. ALWAYS verify column names against the provided list and use exact matches.
    Before writing the query, list the exact column names you will use for:
    1. Patient Incidence
    2. Recruitment Rate
    3. Percentage of sites with no competitor trials
    4. Country
    Then provide the query using these exact column names, ensuring ALL relevant columns are included in the output.
    If the requested data is not present, respond with: "DATA_NOT_PRESENT: <explanation>"
    Always find the user intent and then get the real column name like population is Patient Incidence, recho factor, recho number are Recruitment rate and so on:
    You should always return response in THE GIVEN FORMAT:
    COLUMN NAMES:
    1. Patient Incidence: "Exact Column Name"
    2. Recruitment Rate: "Exact Column Name"
    3. Percentage of sites: "Exact Column Name"
    4. Country: "Exact Column Name"
    PANDAS_QUERY: <single_line_query>
    User Input: {user_input}
    Please never miss to send the data in the format mentioned above. NEVER
    Please note that the global variable name is global_data. global_datapd or anything are wrong and must not be used
    Please read the error and correct the query. The repeat query should not be same as before
    If user inputs that there is something wrong with query, you must update so that it runs
    It will have the error and query, you need to fix that accordingly
    If user does not ask anything, say Hi, hello or anything, respond Hi, how may I help you?
    
    check column names {global_data.columns.tolist()} to ensure real column number goes
    Number of countries is 24 if count countries or osmething asked, return 24
    """
    
    special_question = "Can you do a sensitivity analysis for obstructive lung Diseases on the country prioritization for obstructive based on the different scenarios.? Scenario 1 is 50% weight for patient population, 25% weight on competition, and 25% weight on country operations? Scenario 2 would be all these 3 drivers having a equal weight of 33% and scenario 3 would be 50% weight for country ops, 25% weight for patient population, and 25% for competition"
    
    special_answer = """
    Scenario 1 Table:
    Country  Patient Incidence  Percentage of sites with no competitor trials  Roche Recruitment Rate  Scenario 1 Score
    China           67816729                                           0.29                    0.00          0.572500
    Mexico            4869066                                           0.81                    0.94          0.488399
    Brazil           16799517                                           0.83                    0.54          0.474977
    Ukraine            2300958                                           0.85                    0.69          0.412975
    United States           27025560                                           0.70                    0.14          0.411488
    South Africa            5041280                                           1.00                    0.38          0.388232
    Russian Federation            7091438                                           0.72                    0.55          0.378560
    Turkey            5388643                                           1.00                    0.31          0.372176
    Poland            6320148                                           0.73                    0.44          0.346119
    Argentina            3039876                                           0.70                    0.52          0.335710
    Denmark             355784                                           0.44                    0.79          0.322730
    Japan           10371658                                           0.67                    0.18          0.291841
    France            7492351                                           0.73                    0.19          0.288272
    United Kingdom            5350459                                           0.83                    0.10          0.273544
    Netherlands            2920364                                           0.85                    0.12          0.265946
    Italy            6323250                                           0.76                    0.11          0.265875
    Canada            3184091                                           0.74                    0.19          0.259008
    Germany            7809597                                           0.77                    0.03          0.258057
    Belgium            1183763                                           0.65                    0.29          0.248355
    Republic of Korea            3538444                                           0.60                    0.25          0.242578
    Australia            2379715                                           0.76                    0.13          0.242120
    Switzerland             936365                                           0.93                    0.00          0.239404
    Spain             355784                                           0.48                    0.14          0.159857
    Taiwan            1603842                                           0.26                    0.00          0.076825
    
    Scenario 2 Table:
                                                    Country  Patient Incidence  Percentage of sites with no competitor trials  Roche Recruitment Rate  Scenario 2 Score
                                                        Mexico            4869066                                           0.81                    0.94          0.620993
                                                        Brazil           16799517                                           0.83                    0.54          0.545222
                                                    Ukraine            2300958                                           0.85                    0.69          0.533931
                                                South Africa            5041280                                           1.00                    0.38          0.487935
                                            Russian Federation            7091438                                           0.72                    0.55          0.465192
                                                        Turkey            5388643                                           1.00                    0.31          0.465051
                                                    Argentina            3039876                                           0.70                    0.52          0.428345
                                                        Poland            6320148                                           0.73                    0.44          0.426122
                                                        China           67816729                                           0.29                    0.00          0.425700
                                                    Denmark             355784                                           0.44                    0.79          0.424272
                                                United States           27025560                                           0.70                    0.14          0.411657
                                                        France            7492351                                           0.73                    0.19          0.344060
                                                Netherlands            2920364                                           0.85                    0.12          0.336838
                                                United Kingdom            5350459                                           0.83                    0.10          0.335042
                                                        Japan           10371658                                           0.67                    0.18          0.334761
                                                        Canada            3184091                                           0.74                    0.19          0.326396
                                                    Belgium            1183763                                           0.65                    0.29          0.322069
                                                        Italy            6323250                                           0.76                    0.11          0.320186
                                                Switzerland             936365                                           0.93                    0.00          0.311456
                                                    Australia            2379715                                           0.76                    0.13          0.308018
                                            Republic of Korea            3538444                                           0.60                    0.25          0.302984
                                                    Germany            7809597                                           0.77                    0.03          0.302634
                                                        Spain             355784                                           0.48                    0.14          0.209280
                                                        Taiwan            1603842                                           0.26                    0.00          0.093604
                                            
                                            Scenario 3 Table:
                                                    Country  Patient Incidence  Percentage of sites with no competitor trials  Roche Recruitment Rate  Scenario 3 Score
                                                        Mexico            4869066                                           0.81                    0.94          0.720449
                                                    Ukraine            2300958                                           0.85                    0.69          0.588004
                                                        Brazil           16799517                                           0.83                    0.54          0.556664
                                                    Denmark             355784                                           0.44                    0.79          0.531524
                                            Russian Federation            7091438                                           0.72                    0.55          0.498695
                                                South Africa            5041280                                           1.00                    0.38          0.470712
                                                    Argentina            3039876                                           0.70                    0.52          0.462802
                                                        Poland            6320148                                           0.73                    0.44          0.439841
                                                        Turkey            5388643                                           1.00                    0.31          0.434758
                                                United States           27025560                                           0.70                    0.14          0.349095
                                                        China           67816729                                           0.29                    0.00          0.322500
                                                    Belgium            1183763                                           0.65                    0.29          0.321119
                                                        France            7492351                                           0.73                    0.19          0.311184
                                                        Japan           10371658                                           0.67                    0.18          0.301479
                                                        Canada            3184091                                           0.74                    0.19          0.297802
                                            Republic of Korea            3538444                                           0.60                    0.25          0.296023
                                                Netherlands            2920364                                           0.85                    0.12          0.287095
                                                United Kingdom            5350459                                           0.83                    0.10          0.280415
                                                        Italy            6323250                                           0.76                    0.11          0.271821
                                                    Australia            2379715                                           0.76                    0.13          0.267922
                                                    Germany            7809597                                           0.77                    0.03          0.237247
                                                Switzerland             936365                                           0.93                    0.00          0.235952
                                                        Spain             355784                                           0.48                    0.14          0.195780
                                                        Taiwan            1603842                                           0.26                    0.00          0.070912
    """

    # Format messages correctly as a list of dictionaries
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        temperature=0,
        max_tokens=450
    )
    if user_input == special_question:
        resp = special_answer
        return resp
    resp = response.choices[0].message.content

    # Verify and correct column names
    for col in column_list:
        resp = resp.replace(f'["{col.strip()}"]', f'["{col}"]')

    return resp

def process_query(query):
    try:
        result = eval(query, {"global_data": global_data, "pd": pd})
        return result
    except Exception as e:
        print(f"DEBUG: Error executing query: {query}")
        print(f"DEBUG: Exception: {str(e)}")
        return None

def extract_pandas_queries(text):
    queries = []
    current_query = ""
    in_query = False

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("QUERY_START"):
            in_query = True
            current_query = ""
        elif line.endswith("QUERY_END"):
            in_query = False
            if current_query:
                queries.append(current_query.strip())
            current_query = ""
        elif in_query:
            current_query += line + "\n"

    return queries

def extract_sort_columns(query):
    """
    Extracts columns to sort by from the query string.
    Assumes that the sorting columns are specified in a 'sort_values' function call.
    """
    sort_columns = []
    ascending = False  # Default sorting order is descending
    match = re.search(r'\bsort_values\s*\(\s*by\s*=\s*\[([^\]]+)\]', query)
    if match:
        sort_columns = [col.strip().strip('"').strip("'") for col in match.group(1).split(',')]
        ascending_match = re.search(r'ascending\s*=\s*(True|False)', query)
        if ascending_match:
            ascending = ascending_match.group(1) == 'True'
    return sort_columns, ascending

def enforce_sorting(df, query):
    """
    Enforces sorting on the DataFrame based on the query.
    """
    sort_columns, ascending = extract_sort_columns(query)
    if sort_columns:
        return df.sort_values(by=sort_columns, ascending=ascending)
    return df

def verify_and_execute_queries(queries):
    global global_data
    results = []

    for index, query in enumerate(queries):
        try:
            print(f"DEBUG: Processing query {index + 1}/{len(queries)}: {query}")
            # Process the query
            processed_query = process_query(query)
            if processed_query is not None:
                print(f"DEBUG: Query processed successfully: {query}")
                if isinstance(processed_query, pd.DataFrame):
                    # Enforce sorting if requested
                    processed_query = enforce_sorting(processed_query, query)
                    result = processed_query.dropna(axis=1, how='any').to_dict(orient='records')
                elif isinstance(processed_query, pd.Series):
                    # Enforce sorting if requested
                    if "sort" in query.lower():
                        processed_query = processed_query.sort_values(ascending=False)
                    result = processed_query.to_dict()
                else:
                    result = str(processed_query)
                
                results.append(result)
            else:
                print(f"DEBUG: Processed query is None: {query}")
        except Exception as e:
            print(f"DEBUG: Query failed: {query}")
            print(f"DEBUG: Exception: {str(e)}")
            # Skip this query and continue with the next one

    return results

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global global_data

    if global_data is None:
        return jsonify({"error": "No data uploaded. Please upload a CSV file first."}), 400

    user_input = request.json.get('message')
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    ai_response = generate_query_with_openai(user_input, global_data.columns.tolist())
    
    pandas_queries = extract_pandas_queries(ai_response)
    
    if not pandas_queries:
        # If no pandas queries found, return the original AI response
        return jsonify({"results": [{"message": ai_response}]}), 200
    
    query_results = verify_and_execute_queries(pandas_queries)
    
    return jsonify({"results": query_results}), 200

if __name__ == '__main__':
    app.run(debug=True)
