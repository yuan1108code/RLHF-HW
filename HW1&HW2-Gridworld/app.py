import os
import random
import numpy as np
import time
from flask import Flask, render_template, request, jsonify, redirect, url_for

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# Global variables to store grid state
grid_size = 5
start_position = None
goal_position = None
dead_position = None
obstacles = []
policy = None
values = None
solution_path = []

# Define rewards
GOAL_REWARD = 20
DEAD_PENALTY = -20
OBSTACLE_PENALTY = -1
STEP_PENALTY = -0.1

# Define possible actions
ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Left, Down, Right, Up
ACTION_NAMES = ["left", "down", "right", "up"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/initialize_grid', methods=['POST'])
def initialize_grid():
    global grid_size, start_position, goal_position, dead_position, obstacles, policy, values, solution_path
    
    # Get grid size from request
    data = request.get_json()
    try:
        n = int(data.get('size', 5))
        n = max(5, min(9, n))  # Ensure n is between 5 and 9
    except (TypeError, ValueError):
        n = 5  # Default size
    
    grid_size = n
    start_position = None
    goal_position = None
    dead_position = None
    obstacles = []
    solution_path = []
    
    # Initialize empty policy and values
    policy = [[random.randint(0, 3) for _ in range(n)] for _ in range(n)]
    values = [[0.0 for _ in range(n)] for _ in range(n)]
    
    # For HW1-1, we don't need to randomly generate goal and dead positions
    # Let the user set them instead
    
    return jsonify({
        'size': n,
        'goal': goal_position,
        'dead': dead_position,
        'policy': policy
    })

@app.route('/set_start', methods=['POST'])
def set_start():
    global start_position
    
    data = request.get_json()
    row = data.get('row')
    col = data.get('col')
    
    # Validate position
    if 0 <= row < grid_size and 0 <= col < grid_size:
        # Check if the position is not already assigned
        if (row, col) != goal_position and (row, col) != dead_position and (row, col) not in obstacles:
            start_position = (row, col)
            return jsonify({'success': True, 'start': start_position})
    
    return jsonify({'success': False})

@app.route('/toggle_obstacle', methods=['POST'])
def toggle_obstacle():
    global obstacles
    
    data = request.get_json()
    row = data.get('row')
    col = data.get('col')
    
    # Validate position
    if 0 <= row < grid_size and 0 <= col < grid_size:
        position = (row, col)
        
        # Check if the position is not already assigned to other elements
        if position != start_position and position != goal_position and position != dead_position:
            if position in obstacles:
                obstacles.remove(position)
                return jsonify({'success': True, 'action': 'removed', 'position': position})
            else:
                if len(obstacles) < grid_size - 2:  # Ensure we have at most n-2 obstacles
                    obstacles.append(position)
                    return jsonify({'success': True, 'action': 'added', 'position': position})
                else:
                    return jsonify({'success': False, 'message': '已達到最大障礙物數量'})
    
    return jsonify({'success': False})

@app.route('/evaluate_policy', methods=['POST'])
def evaluate_policy():
    global policy, values, solution_path
    
    # Check if grid is fully initialized
    if start_position is None:
        return jsonify({'success': False, 'message': '請先設置起始位置'})
    
    if goal_position is None:
        return jsonify({'success': False, 'message': '請先設置目標位置'})
    
    # Create the grid with rewards
    rewards = [[STEP_PENALTY for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Set special cell rewards
    rewards[goal_position[0]][goal_position[1]] = GOAL_REWARD
    
    # For HW1-1, we might not have a dead position
    if dead_position:
        rewards[dead_position[0]][dead_position[1]] = DEAD_PENALTY
    
    # Set obstacle penalties
    for obs in obstacles:
        rewards[obs[0]][obs[1]] = OBSTACLE_PENALTY
    
    # Define terminal states
    is_terminal = [[False for _ in range(grid_size)] for _ in range(grid_size)]
    is_terminal[goal_position[0]][goal_position[1]] = True
    
    if dead_position:
        is_terminal[dead_position[0]][dead_position[1]] = True
    
    # Policy evaluation
    gamma = 0.9  # discount factor
    theta = 0.0001  # convergence threshold
    
    # Initialize value function
    values = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Policy evaluation algorithm
    iteration = 0
    max_iterations = 1000
    
    while iteration < max_iterations:
        delta = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if is_terminal[i][j] or (i, j) in obstacles:
                    continue
                
                old_value = values[i][j]
                
                # Get the action from policy
                action_idx = policy[i][j]
                action = ACTIONS[action_idx]
                
                # Compute next state
                next_i = i + action[0]
                next_j = j + action[1]
                
                # Check if next state is valid
                if 0 <= next_i < grid_size and 0 <= next_j < grid_size:
                    # If next state is an obstacle, agent stays in place
                    if (next_i, next_j) in obstacles:
                        values[i][j] = rewards[i][j] + gamma * values[i][j]
                    else:
                        values[i][j] = rewards[next_i][next_j] + gamma * values[next_i][next_j]
                else:
                    # If next state is outside the grid, agent stays in place
                    values[i][j] = rewards[i][j] + gamma * values[i][j]
                
                delta = max(delta, abs(old_value - values[i][j]))
        
        if delta < theta:
            break
        
        iteration += 1
    
    # Format values for display
    formatted_values = [[round(val, 2) for val in row] for row in values]
    
    # Calculate the optimal path from start to goal
    solution_path = calculate_path(start_position, goal_position)
    
    return jsonify({
        'success': True,
        'values': formatted_values,
        'policy': policy,
        'iterations': iteration,
        'path': solution_path
    })

def calculate_path(start, goal):
    """Calculate the optimal path from start to goal using the current policy"""
    path = [start]
    current = start
    max_steps = grid_size * grid_size
    step_count = 0
    
    while current != goal and step_count < max_steps:
        i, j = current
        action_idx = policy[i][j]
        action = ACTIONS[action_idx]
        
        next_i = i + action[0]
        next_j = j + action[1]
        
        # Check if next state is valid
        if 0 <= next_i < grid_size and 0 <= next_j < grid_size and (next_i, next_j) not in obstacles:
            current = (next_i, next_j)
            path.append(current)
        else:
            # If we hit an obstacle or boundary, we should follow the policy more intelligently
            # Try all actions and pick the one with highest value
            best_val = float('-inf')
            best_next = None
            
            for a_idx, a in enumerate(ACTIONS):
                n_i = i + a[0]
                n_j = j + a[1]
                
                if 0 <= n_i < grid_size and 0 <= n_j < grid_size and (n_i, n_j) not in obstacles:
                    if values[n_i][n_j] > best_val:
                        best_val = values[n_i][n_j]
                        best_next = (n_i, n_j)
            
            if best_next:
                current = best_next
                path.append(current)
            else:
                # No valid moves available
                break
        
        # Check if we've reached a terminal state
        if current == goal or (dead_position and current == dead_position):
            break
            
        step_count += 1
    
    return path

@app.route('/randomize_policy', methods=['POST'])
def randomize_policy():
    global policy
    
    # Generate random policy
    policy = [[random.randint(0, 3) for _ in range(grid_size)] for _ in range(grid_size)]
    
    return jsonify({
        'success': True,
        'policy': policy
    })

@app.route('/get_solution_path', methods=['GET'])
def get_solution_path():
    return jsonify({'path': solution_path})


@app.route('/set_goal', methods=['POST'])
def set_goal():
    global goal_position
    
    data = request.get_json()
    row = data.get('row')
    col = data.get('col')
    
    # Validate position
    if 0 <= row < grid_size and 0 <= col < grid_size:
        # Check if the position is not already assigned
        if (row, col) != start_position and (row, col) not in obstacles:
            goal_position = (row, col)
            return jsonify({'success': True, 'goal': goal_position})
    
    return jsonify({'success': False})

@app.route('/value_iteration', methods=['POST'])
def value_iteration():
    global policy, values, solution_path
    
    # Check if grid is fully initialized
    if start_position is None:
        return jsonify({'success': False, 'message': '請先設置起始位置'})
    
    if goal_position is None:
        return jsonify({'success': False, 'message': '請先設置目標位置'})
    
    # Create the grid with rewards
    rewards = [[STEP_PENALTY for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Set special cell rewards
    rewards[goal_position[0]][goal_position[1]] = GOAL_REWARD
    
    # For HW1-1, we might not have a dead position
    if dead_position:
        rewards[dead_position[0]][dead_position[1]] = DEAD_PENALTY
    
    # Set obstacle penalties
    for obs in obstacles:
        rewards[obs[0]][obs[1]] = OBSTACLE_PENALTY
    
    # Define terminal states
    is_terminal = [[False for _ in range(grid_size)] for _ in range(grid_size)]
    is_terminal[goal_position[0]][goal_position[1]] = True
    
    if dead_position:
        is_terminal[dead_position[0]][dead_position[1]] = True
    
    # Value iteration parameters
    gamma = 0.99  # discount factor (matching reference notebook)
    theta = 0.001  # convergence threshold (matching reference notebook)
    
    # Initialize value function
    values = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Value iteration algorithm
    iteration = 0
    max_iterations = 1000
    
    while iteration < max_iterations:
        delta = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if is_terminal[i][j] or (i, j) in obstacles:
                    continue
                
                old_value = values[i][j]
                
                # Initialize max value for this state
                max_value = float('-inf')
                
                # Try all possible actions
                for action_idx, action in enumerate(ACTIONS):
                    # Calculate value for this action
                    next_i = i + action[0]
                    next_j = j + action[1]
                    
                    # Check if next state is valid
                    if 0 <= next_i < grid_size and 0 <= next_j < grid_size and (next_i, next_j) not in obstacles:
                        # Calculate expected value with stochastic transitions (like in reference notebook)
                        # Main direction (80% probability)
                        value = 0.8 * (rewards[next_i][next_j] + gamma * values[next_i][next_j])
                        
                        # Left of intended direction (10% probability)
                        left_action = ACTIONS[(action_idx - 1) % 4]
                        left_i = i + left_action[0]
                        left_j = j + left_action[1]
                        
                        if 0 <= left_i < grid_size and 0 <= left_j < grid_size and (left_i, left_j) not in obstacles:
                            value += 0.1 * (rewards[left_i][left_j] + gamma * values[left_i][left_j])
                        else:
                            # If hitting boundary or obstacle, stay in place
                            value += 0.1 * (rewards[i][j] + gamma * values[i][j])
                        
                        # Right of intended direction (10% probability)
                        right_action = ACTIONS[(action_idx + 1) % 4]
                        right_i = i + right_action[0]
                        right_j = j + right_action[1]
                        
                        if 0 <= right_i < grid_size and 0 <= right_j < grid_size and (right_i, right_j) not in obstacles:
                            value += 0.1 * (rewards[right_i][right_j] + gamma * values[right_i][right_j])
                        else:
                            # If hitting boundary or obstacle, stay in place
                            value += 0.1 * (rewards[i][j] + gamma * values[i][j])
                    else:
                        # If hitting boundary or obstacle, stay in place
                        value = rewards[i][j] + gamma * values[i][j]
                    
                    # Update max value if this action is better
                    max_value = max(max_value, value)
                
                # Update value function with the best action's value
                values[i][j] = max_value
                
                # Update delta for convergence check
                delta = max(delta, abs(old_value - values[i][j]))
        
        if delta < theta:
            break
        
        iteration += 1
    
    # Extract policy from value function
    for i in range(grid_size):
        for j in range(grid_size):
            if is_terminal[i][j] or (i, j) in obstacles:
                continue
            
            # Find the best action for this state
            best_action = 0
            best_value = float('-inf')
            
            for action_idx, action in enumerate(ACTIONS):
                next_i = i + action[0]
                next_j = j + action[1]
                
                if 0 <= next_i < grid_size and 0 <= next_j < grid_size and (next_i, next_j) not in obstacles:
                    # Calculate expected value with stochastic transitions
                    value = 0.8 * (rewards[next_i][next_j] + gamma * values[next_i][next_j])
                    
                    # Left of intended direction
                    left_action = ACTIONS[(action_idx - 1) % 4]
                    left_i = i + left_action[0]
                    left_j = j + left_action[1]
                    
                    if 0 <= left_i < grid_size and 0 <= left_j < grid_size and (left_i, left_j) not in obstacles:
                        value += 0.1 * (rewards[left_i][left_j] + gamma * values[left_i][left_j])
                    else:
                        value += 0.1 * (rewards[i][j] + gamma * values[i][j])
                    
                    # Right of intended direction
                    right_action = ACTIONS[(action_idx + 1) % 4]
                    right_i = i + right_action[0]
                    right_j = j + right_action[1]
                    
                    if 0 <= right_i < grid_size and 0 <= right_j < grid_size and (right_i, right_j) not in obstacles:
                        value += 0.1 * (rewards[right_i][right_j] + gamma * values[right_i][right_j])
                    else:
                        value += 0.1 * (rewards[i][j] + gamma * values[i][j])
                else:
                    value = rewards[i][j] + gamma * values[i][j]
                
                if value > best_value:
                    best_value = value
                    best_action = action_idx
            
            policy[i][j] = best_action
    
    # Format values for display
    formatted_values = [[round(val, 2) for val in row] for row in values]
    
    # Calculate the optimal path from start to goal
    solution_path = calculate_path(start_position, goal_position)
    
    return jsonify({
        'success': True,
        'values': formatted_values,
        'policy': policy,
        'iterations': iteration,
        'path': solution_path
    })

@app.route('/set_dead', methods=['POST'])
def set_dead():
    global dead_position
    
    data = request.get_json()
    row = data.get('row')
    col = data.get('col')
    
    # Validate position
    if 0 <= row < grid_size and 0 <= col < grid_size:
        # Check if the position is not already assigned
        if (row, col) != start_position and (row, col) != goal_position and (row, col) not in obstacles:
            dead_position = (row, col)
            return jsonify({'success': True, 'dead': dead_position})
    
    return jsonify({'success': False})

@app.route('/simulate_learning', methods=['POST'])
def simulate_learning():
    global policy, values, solution_path
    
    # Check if grid is fully initialized
    if start_position is None:
        return jsonify({'success': False, 'message': '請先設置起始位置'})
    
    if goal_position is None:
        return jsonify({'success': False, 'message': '請先設置目標位置'})
    
    # Create the grid with rewards
    rewards = [[STEP_PENALTY for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Set special cell rewards
    rewards[goal_position[0]][goal_position[1]] = GOAL_REWARD
    
    if dead_position:
        rewards[dead_position[0]][dead_position[1]] = DEAD_PENALTY
    
    # Set obstacle penalties
    for obs in obstacles:
        rewards[obs[0]][obs[1]] = OBSTACLE_PENALTY
    
    # Define terminal states
    is_terminal = [[False for _ in range(grid_size)] for _ in range(grid_size)]
    is_terminal[goal_position[0]][goal_position[1]] = True
    
    if dead_position:
        is_terminal[dead_position[0]][dead_position[1]] = True
    
    # Generate a series of learning steps to simulate the learning process
    learning_steps = []
    
    # Start with random policy
    current_policy = [[random.randint(0, 3) for _ in range(grid_size)] for _ in range(grid_size)]
    current_values = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Calculate initial path (will likely be random and inefficient)
    initial_path = []
    try:
        # Set the global policy temporarily to calculate path
        temp_policy = policy
        policy = current_policy
        temp_values = values
        values = current_values
        
        initial_path = calculate_path(start_position, goal_position)
        
        # Restore global policy
        policy = temp_policy
        values = temp_values
    except Exception as e:
        print(f"Error calculating initial path: {e}")
        initial_path = []  # If path calculation fails, use empty path
    
    learning_steps.append({
        'policy': [row[:] for row in current_policy],
        'values': [[0.0 for _ in range(grid_size)] for _ in range(grid_size)],
        'path': initial_path,
        'message': f'初始隨機策略: 路徑長度 {len(initial_path) if initial_path else "無法到達目標"}'
    })
    
    # Simulate several iterations of policy improvement
    gamma = 0.9  # discount factor
    
    for iteration in range(5):  # Simulate 5 learning iterations
        # Evaluate current policy
        current_values = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]
        
        # Policy evaluation
        delta = float('inf')
        theta = 0.0001  # Convergence threshold
        eval_iterations = 0
        max_eval_iterations = 100  # Prevent infinite loops
        
        while delta > theta and eval_iterations < max_eval_iterations:
            delta = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if is_terminal[i][j] or (i, j) in obstacles:
                        continue
                    
                    old_value = current_values[i][j]
                    
                    # Get action from policy
                    action_idx = current_policy[i][j]
                    action = ACTIONS[action_idx]
                    
                    # Calculate next state
                    next_i = i + action[0]
                    next_j = j + action[1]
                    
                    # Check if next state is valid
                    if 0 <= next_i < grid_size and 0 <= next_j < grid_size and (next_i, next_j) not in obstacles:
                        current_values[i][j] = rewards[next_i][next_j] + gamma * current_values[next_i][next_j]
                    else:
                        # If hitting boundary or obstacle, stay in place
                        current_values[i][j] = rewards[i][j] + gamma * current_values[i][j]
                    
                    delta = max(delta, abs(old_value - current_values[i][j]))
            
            eval_iterations += 1
        
        # Calculate path for this iteration
        current_path = []
        try:
            # Set the global policy temporarily to calculate path
            temp_policy = policy
            policy = current_policy
            temp_values = values
            values = current_values
            
            current_path = calculate_path(start_position, goal_position)
            
            # Restore global policy
            policy = temp_policy
            values = temp_values
        except Exception as e:
            print(f"Error calculating path for iteration {iteration}: {e}")
        
        # Add this step to learning steps
        learning_steps.append({
            'policy': [row[:] for row in current_policy],  # Deep copy
            'values': [[round(val, 2) for val in row] for row in current_values],
            'path': current_path,
            'message': f'學習迭代 {iteration+1}: 路徑長度 {len(current_path) if current_path else "無法到達目標"}'
        })
        
        # Improve policy based on current values
        improved = False
        for i in range(grid_size):
            for j in range(grid_size):
                if is_terminal[i][j] or (i, j) in obstacles:
                    continue
                
                # Find best action
                best_action = 0
                best_value = float('-inf')
                
                for action_idx, action in enumerate(ACTIONS):
                    next_i = i + action[0]
                    next_j = j + action[1]
                    
                    if 0 <= next_i < grid_size and 0 <= next_j < grid_size and (next_i, next_j) not in obstacles:
                        value = rewards[next_i][next_j] + gamma * current_values[next_i][next_j]
                    else:
                        # If hitting boundary or obstacle, stay in place
                        value = rewards[i][j] + gamma * current_values[i][j]
                    
                    if value > best_value:
                        best_value = value
                        best_action = action_idx
                
                # Update policy if better action found
                if best_action != current_policy[i][j]:
                    current_policy[i][j] = best_action
                    improved = True
        
        # If policy didn't improve, we're done
        if not improved:
            break
    
    # Calculate final path
    final_path = []
    try:
        # Set the global policy temporarily to calculate path
        temp_policy = policy
        policy = current_policy
        temp_values = values
        values = current_values
        
        final_path = calculate_path(start_position, goal_position)
        
        # Restore global policy
        policy = temp_policy
        values = temp_values
    except Exception as e:
        print(f"Error calculating final path: {e}")
    
    # Add final step
    learning_steps.append({
        'policy': [row[:] for row in current_policy],
        'values': [[round(val, 2) for val in row] for row in current_values],
        'path': final_path,
        'message': f'最終策略: 路徑長度 {len(final_path) if final_path else "無法到達目標"}'
    })
    
    # Update global variables with final results
    policy = current_policy
    values = current_values
    solution_path = final_path
    
    return jsonify({
        'success': True,
        'learning_steps': learning_steps,
        'auto_start': True  # Signal to automatically start the animation
    })

@app.route('/auto_setup', methods=['POST'])
def auto_setup():
    global grid_size, start_position, goal_position, dead_position, obstacles, policy, values, solution_path
    
    # Get grid size from request
    data = request.get_json()
    try:
        n = int(data.get('size', 5))
        n = max(5, min(9, n))  # Ensure n is between 5 and 9
    except (TypeError, ValueError):
        n = 5  # Default size
    
    grid_size = n
    
    # Clear previous settings
    obstacles = []
    solution_path = []
    
    # Automatically set start position (top-left corner)
    start_position = (0, 0)
    
    # Automatically set goal position (bottom-right corner)
    goal_position = (n-1, n-1)
    
    # Automatically set dead position (if grid is large enough)
    if n >= 5:
        dead_position = (n//2, n//2)
    else:
        dead_position = None
    
    # Randomly place obstacles (avoiding start, goal, and dead positions)
    max_obstacles = n - 2  # Maximum number of obstacles
    num_obstacles = random.randint(1, max_obstacles)
    
    for _ in range(num_obstacles):
        while True:
            row = random.randint(0, n-1)
            col = random.randint(0, n-1)
            
            # Make sure obstacle doesn't overlap with special positions
            if (row, col) != start_position and (row, col) != goal_position and (row, col) != dead_position and (row, col) not in obstacles:
                obstacles.append((row, col))
                break
    
    # Initialize random policy and values
    policy = [[random.randint(0, 3) for _ in range(n)] for _ in range(n)]
    values = [[0.0 for _ in range(n)] for _ in range(n)]
    
    return jsonify({
        'size': n,
        'start': start_position,
        'goal': goal_position,
        'dead': dead_position,
        'obstacles': obstacles,
        'policy': policy
    })

if __name__ == '__main__':
    app.run(debug=True)