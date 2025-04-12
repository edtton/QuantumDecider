import pytesseract
from PIL import Image
import re
import google.generativeai as genai
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
import json
import time 

genai.configure(api_key="AIzaSyDn6WqMdfh0tojInAmpAAVf5Ms2BI-jPIM")

IMAGE_DIRECTORY = "static/images"  # Change this to your desired directory

app = Flask(__name__)
app.secret_key = 'bitcamp2025' 

os.makedirs(IMAGE_DIRECTORY, exist_ok=True)
app.config['IMAGE_DIRECTORY'] = IMAGE_DIRECTORY

@app.route('/')
def landing():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        image = request.files['image']
        user_prompt = request.form.get('user_prompt') 

        if image:
            filepath = os.path.join(app.config['IMAGE_DIRECTORY'], image.filename)
            image.save(filepath)
            session['image_filename'] = image.filename

            try:
                print(f"\nProcessing image: {image.filename}")
                print("-" * 50)

                menu_dict = extract_menu_from_image(filepath, user_prompt)
                print(menu_dict)

                if menu_dict:
                    print("\nMenu Items and Prices:")
                    print("-" * 50)
                    for item, price in menu_dict.items():
                        print(f"{item}: ${float(price):.2f}")
                    print("-" * 50)

                    num_items = menu_dict.__len__()
                    selection_result = make_selection(num_items)
                    
                    selected_item = list(menu_dict)[selection_result['index']]
                    session['selected_item'] = selected_item
                    session['menu_dict'] = menu_dict
                    session['user_prompt'] = user_prompt
                    session['quantum_time'] = selection_result['quantum_time']
                    session['classical_time'] = selection_result['classical_time']
                    session['quantum_counts'] = selection_result['quantum_counts']
                    session['optimal_iterations'] = selection_result['optimal_iterations']
                    session['num_qubits'] = selection_result['num_qubits']
                    session['total_shots'] = selection_result['total_shots']

                    return redirect(url_for('results'))
                else:
                    return "No menu items could be extracted from this image."

            except Exception as e:
                return f"An error occurred: {str(e)}"
        else:
            return "No image was uploaded."

    return render_template('upload.html')

@app.route('/results')
def results():
    selected_item = session.get('selected_item')
    menu_dict = session.get('menu_dict', {})
    user_prompt = session.get('user_prompt', 'No prompt provided.')
    quantum_time = session.get('quantum_time')
    classical_time = session.get('classical_time')
    quantum_counts = session.get('quantum_counts', {})
    optimal_iterations = session.get('optimal_iterations', 0)

    # Calculate quantum speedup
    if quantum_time and classical_time:
        speedup = classical_time / quantum_time
    else:
        speedup = 0

    return render_template('results.html',
                         selected_item=selected_item,
                         menu=menu_dict,
                         prompt=user_prompt,
                         quantum_time=quantum_time,
                         classical_time=classical_time,
                         speedup=speedup,
                         quantum_counts=quantum_counts,
                         optimal_iterations=optimal_iterations)

@app.route('/choices', methods=['GET', 'POST'])
def choices():
    if request.method == 'POST':
        choices = request.form.getlist('choices')

        if not choices:
            return "No choices submitted."

        num_items = len(choices)
        selection_result = make_selection(num_items)
        selected_item = choices[selection_result['index']]

        session['selected_item'] = selected_item
        session['menu_dict'] = {choice: 'N/A' for choice in choices}
        session['user_prompt'] = "User entered custom choices"
        session['quantum_time'] = selection_result['quantum_time']
        session['classical_time'] = selection_result['classical_time']

        return redirect(url_for('results'))

    return render_template('choices.html')

def clean_json_string(text):
    """Clean and extract JSON from the response text."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        json_str = json_str.replace('```json', '').replace('```', '')
        return json_str
    return text

def extract_menu_from_image(image_path, user_prompt):
    try:
        image = Image.open(image_path)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
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
        response = model.generate_content([full_prompt, image])
        json_str = clean_json_string(response.text)
        
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

def diffusion_operator(n):
    """Create the diffusion operator for Grover's algorithm."""
    diffuser = QuantumCircuit(n)
    diffuser.h(range(n))
    diffuser.x(range(n))
    diffuser.h(n - 1)
    diffuser.mcx(list(range(n - 1)), n - 1)
    diffuser.h(n - 1)
    diffuser.x(range(n))
    diffuser.h(range(n))
    return diffuser

def make_selection(num_items):
    # Quantum Selection using Grover's Algorithm
    quantum_start = time.time()
    
    # Calculate required qubits and circuit parameters
    num_qubits = ceil(log2(num_items))
    search_space_size = 2 ** num_qubits
    optimal_iterations = math.floor((math.pi / 4) * math.sqrt(search_space_size))
    
    # Create random target (this simulates what we're searching for)
    target = random.randint(0, num_items - 1)
    target_bin = format(target, f'0{num_qubits}b')
    
    # Initialize the quantum circuit
    grover_circuit = QuantumCircuit(num_qubits, num_qubits)
    grover_circuit.h(range(num_qubits))  # Initialize superposition
    
    # Create oracle for the target state
    oracle = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        if target_bin[i] == '0':
            oracle.x(i)
    oracle.h(num_qubits - 1)
    oracle.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    oracle.h(num_qubits - 1)
    
    # Apply Grover iteration (oracle + diffusion)
    grover_circuit.append(oracle.to_gate(), range(num_qubits))
    grover_circuit.append(diffusion_operator(num_qubits).to_gate(), range(num_qubits))
    
    # Measure the results
    grover_circuit.measure(range(num_qubits), range(num_qubits))
    
    # Execute the circuit with multiple shots to get statistics
    simulator = AerSimulator()
    compiled_circuit = transpile(grover_circuit, simulator)
    shots = 1000  # Increase number of shots for better statistics
    result = simulator.run(compiled_circuit, shots=shots).result()
    counts = result.get_counts()
    
    # Get most frequent result
    measured = max(counts, key=counts.get)
    measured_decimal = int(measured, 2)
    quantum_end = time.time()
    quantum_time = quantum_end - quantum_start
    
    # Classical Selection (Linear Search)
    classical_start = time.time()
    found_index = -1
    for i in range(num_items):
        if i == target:
            found_index = i
            break
    classical_end = time.time()
    classical_time = classical_end - classical_start
    
    # Format quantum counts for display
    formatted_counts = {}
    total_shots = sum(counts.values())
    for state, count in counts.items():
        # Calculate percentage and format state
        percentage = (count / total_shots) * 100
        formatted_state = state
        formatted_counts[formatted_state] = {
            'count': count,
            'percentage': percentage,
            'is_solution': state == measured
        }
    
    return {
        'index': measured_decimal % num_items,
        'quantum_time': quantum_time,
        'classical_time': classical_time,
        'quantum_counts': formatted_counts,
        'optimal_iterations': optimal_iterations,
        'num_qubits': num_qubits,
        'total_shots': total_shots
    }

if __name__ == "__main__":
    app.run(debug=True)






