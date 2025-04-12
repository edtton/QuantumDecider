import pytesseract
from PIL import Image
import re
import google.generativeai as genai
from google.genai import types
import PIL.Image
import random 
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import math 
from math import log2, ceil
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile
import os
from pathlib import Path
from PIL import Image
import json
import re
import time 

genai.configure(api_key="AIzaSyDn6WqMdfh0tojInAmpAAVf5Ms2BI-jPIM")

IMAGE_DIRECTORY = "static/images"  # Change this to your desired directory

app = Flask(__name__)
app.secret_key = 'bitcamp2025' 

def diffusion_operator(n):
    diffuser = QuantumCircuit(n)
    diffuser.h(range(n))
    diffuser.x(range(n))
    diffuser.h(n - 1)
    diffuser.mcx(list(range(n - 1)), n - 1)
    diffuser.h(n - 1)
    diffuser.x(range(n))
    diffuser.h(range(n))
    return diffuser

def quantum_search(number, order_type, search_type, target_number):
    dataset_size = number 
    num_qubits = ceil(log2(dataset_size))
    search_space_size = 2 ** num_qubits
    optimal_iterations = math.floor((math.pi / 4) * math.sqrt(search_space_size))
    dataset = list(range(dataset_size))

    if order_type == "random":
        random.shuffle(dataset)

    target = target_number
    target_bin = format(target, f'0{num_qubits}b')

    grover_circuit = QuantumCircuit(num_qubits, num_qubits)

    # Step 1: Initialization - Apply Hadamard gates to all qubits
    grover_circuit.h(range(num_qubits))

    # Step 2: Build the oracle
    oracle = QuantumCircuit(num_qubits)

    for i in range(num_qubits):
        if target_bin[i] == '0':
            oracle.x(i)

    # oracle.cz(0, num_qubits - 1)
    oracle.h(num_qubits - 1)
    oracle.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    oracle.h(num_qubits - 1)

    grover_circuit.append(oracle.to_gate(), range(num_qubits))
    grover_circuit.append(diffusion_operator(num_qubits).to_gate(), range(num_qubits))

    # Step 5: Measure the results
    grover_circuit.measure(range(num_qubits), range(num_qubits))

    # For execution
    simulator = AerSimulator()
    compiled_circuit = transpile(grover_circuit, simulator)

    measured_decimal = -1

    start = time.time()
    # while measured_decimal != target:
    sim_result = simulator.run(compiled_circuit, shots=optimal_iterations).result()
    counts = sim_result.get_counts()
    # measured = list(counts.keys())[0]
    measured = max(counts, key=counts.get)
    measured_decimal = int(measured, 2)
    end = time.time()

    # Output
    # print(counts)
    print(f"Grover Measured Result: {measured_decimal}")
    print(f"Grover Execution Time: {end - start:.6f} seconds")

    if search_type == "linear":
        start = time.time()
        for i in range(dataset_size):
            if dataset[i] == target:
                found = True
                break
        end = time.time()

        print(f"Linear Measured Result: {measured_decimal}")
        print(f"Linear Execution Time: {end - start:.6f} seconds")
    else: 
        print("Invalid search type.")

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         number = int(request.form["number"])
#         order_type = request.form["order_type"]
#         search_type = request.form["search_type"]
#         target_number = int(request.form["target"])

#         # Optional: enforce target_number <= number
#         if target_number > number:
#             return "Target number must be less than or equal to dataset size.", 400

#         # Placeholder for your quantum algorithm
#         result = quantum_search(number, order_type, search_type, target_number)

#         # return redirect(url_for("results", number=number, order_type=order_type, search_type=search_type, target_number=target_number))

#     return render_template("index2.html")

# @app.route("/results")
# def results():
#     number = request.args.get("number")
#     order_type = request.args.get("order_type")
#     search_type = request.args.get("search_type")
#     target_number = request.args.get("target_number")

#     # Placeholder response
#     return f"Received: number={number}, order={order_type}, search={search_type}, target={target_number}"

# if __name__ == "__main__":
#     app.run(debug=True)

@app.route('/')
def landing():
    return render_template('index.html')

os.makedirs(IMAGE_DIRECTORY, exist_ok=True)
app.config['IMAGE_DIRECTORY'] = IMAGE_DIRECTORY

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        image = request.files['image']
        user_prompt = request.form.get('user_prompt') 

        if image:
            filepath = os.path.join(app.config['IMAGE_DIRECTORY'], image.filename)
            image.save(filepath)

        try:
            if not os.path.exists(IMAGE_DIRECTORY):
                os.makedirs(IMAGE_DIRECTORY)

            image_files = [f for f in os.listdir(IMAGE_DIRECTORY) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

            if not image_files:
                return "No images found in the image directory."

            for image_file in image_files:
                image_path = os.path.join(IMAGE_DIRECTORY, image_file)
                print(f"\nProcessing image: {image_file}")
                print("-" * 50)

                menu_dict = extract_menu_from_image(image_path, user_prompt)
                print(menu_dict)

                if menu_dict:
                    print("\nMenu Items and Prices:")
                    print("-" * 50)
                    for item, price in menu_dict.items():
                        print(f"{item}: ${float(price):.2f}")
                    print("-" * 50)

                    num_items = menu_dict.__len__()
                    index = num_items + 1  

                    while index >= num_items:
                        index = make_selection(num_items)

                    selected_item = index
                                        # Store in session
                    session['selected_item'] = list(menu_dict)[selected_item]
                    session['menu_dict'] = menu_dict
                    session['user_prompt'] = user_prompt

                    return redirect(url_for('results'))
                    # print(menu_dict[selected_item])
                    # print(menu_dict[list(menu_dict)[selected_item]])
                    print(selected_item)
                    key = list(menu_dict)[selected_item]
                    return(key)
                else:
                    return("No menu items could be extracted from this image.")

        except Exception as e:
            return f"An error occurred: {str(e)}"

    return render_template('upload.html')

@app.route('/results')
def results():
    selected_item = session.get('selected_item')
    menu_dict = session.get('menu_dict', {})
    user_prompt = session.get('user_prompt', 'No prompt provided.')

    return render_template('results.html',
                           selected_item=selected_item,
                           menu=menu_dict,
                           prompt=user_prompt)


@app.route('/choices', methods=['GET', 'POST'])
def choices():
    if request.method == 'POST':
        choices = request.form.getlist('choices')

        if not choices:
            return "No choices submitted."

        # Simulate selection (same logic as from image route)
        num_items = len(choices)
        index = num_items + 1

        while index >= num_items:
            index = make_selection(num_items)

        selected_item = choices[index]

        # Mimic structure used by /upload
        session['selected_item'] = selected_item
        session['menu_dict'] = {choice: 'N/A' for choice in choices}  # Dummy prices
        session['user_prompt'] = "User entered custom choices"

        return redirect(url_for('results'))

    return render_template('choices.html')


def clean_json_string(text):
    """Clean and extract JSON from the response text."""
    # Find content between curly braces
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        # Remove any markdown formatting
        json_str = json_str.replace('```json', '').replace('```', '')
        return json_str
    return text

def extract_menu_from_image(image_path, user_prompt):
    """
    Extract menu items and prices from an image using Gemini API.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Dictionary with menu items as keys and prices as values
    """
    try:
        # Load the image
        image = Image.open(image_path)
        
        # Initialize the Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create a specific prompt for menu extraction
        prompt = """
        Analyze this menu image and create a JSON object where:
        - Each key is a menu item name (including section if present)
        - Each value is the price as a number (without $ symbol)
        
        Format the response EXACTLY like this example:
        {
            "Appetizers - Spring Rolls": 8.99,
            "Main - Chicken Curry": 15.99
        }
        
        Important:
        1. Only return valid JSON, no other text
        2. Remove the $ symbol from prices if prices are listed
        3. Use numbers for prices, not strings if prices are listed 
        4. include section names where applicable
        5. Make sure all JSON syntax is correct
        6. If the user specifies a type of dish or item below, make sure to only include items that match that description
        7. If there are only item names, assume the price is $0.00
        """

        full_prompt = prompt + "\n" + user_prompt
        
        # Generate content from the image
        response = model.generate_content([full_prompt, image])
        
        # Clean up the response text
        json_str = clean_json_string(response.text)
        
        # Try to parse the response as JSON
        try:
            menu_dict = json.loads(json_str)
            return menu_dict
        except json.JSONDecodeError as e:
            print(f"Error parsing AI response for {image_path}")
            print(f"Response was: {response.text}")
            return {}
    
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return {}


def make_selection(num_items):
    num_qubits = math.ceil(math.log2(num_items))

    # Create circuit with num_qubits and matching classical bits
    qc = QuantumCircuit(num_qubits, num_qubits)

    # place all qubits in superposition
    qc.h(range(num_qubits))  

    # Measure all qubits
    qc.measure(range(num_qubits), range(num_qubits))

    simulator = AerSimulator()
    compiled_circuit = transpile(qc, simulator)
    sim_result = simulator.run(compiled_circuit).result()
    counts = sim_result.get_counts()

    bitstring = list(counts.keys())[0]
    index = int(bitstring, 2)

    return index 






