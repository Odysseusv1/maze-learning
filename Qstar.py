import random
import time
import os

# gridworld settings

ROWS = 3
COLS = 4

# Reward layout for each grid cell
reward_map = [
    [0,   0,  0,  10],    # +10 goal
    [0,   0,  0, -10],    # -10 goal
    [0,   0,  0,   0]     # starting row
]

# terminal positions (episode ends here)
terminal_positions = {(0, 3), (1, 3)}

# blocked cells (cannot enter) can be a wall i guess
blocked_positions = {(1, 1)}

# initialize the q table
q_values = [
    [{"up": 0, "down": 0, "left": 0, "right": 0} for _ in range(COLS)]
    for _ in range(ROWS)
]

start_position = (2, 0)
agent_position = start_position

learning_rate = 0.5
discount_rate = 0.5
exploration_rate = 0.2   # epsilon (chance of random move) idk i dont make the algorithm

# Q-learning functions

def available_actions(position):
    # ts speaks for itself
    row, col = position
    actions = []

    if row > 0:
        actions.append("up")
    if row < ROWS - 1:
        actions.append("down")
    if col > 0:
        actions.append("left")
    if col < COLS - 1:
        actions.append("right")

    return actions


def next_position(position, action):
    row, col = position
    if action == "up":
        return (row - 1, col)
    if action == "down":
        return (row + 1, col)
    if action == "left":
        return (row, col - 1)
    if action == "right":
        return (row, col + 1)


def get_q(position, action):
    row, col = position
    return q_values[row][col][action]


def set_q(position, action, value):
    row, col = position
    q_values[row][col][action] = value


def updated_q_value(position, action):
    # bellman equation
    current_q = get_q(position, action)
    next_pos = next_position(position, action)

    reward = reward_map[next_pos[0]][next_pos[1]]
    best_future_q = max(q_values[next_pos[0]][next_pos[1]].values())

    return current_q + learning_rate * (
        reward + discount_rate * best_future_q - current_q
    )


def choose_action(position):
    #Choose an action using epsilon-greedy exploration
    actions = available_actions(position)

    if random.random() < exploration_rate:
        return random.choice(actions)

    # choose action with highest Q-value
    best_action = None
    best_q = float("-inf")
    for a in actions:
        q = get_q(position, a)
        if q > best_q:
            best_q = q
            best_action = a
    return best_action

# display functions

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def draw_gridworld(agent_pos):
    # display the stuff (agent, walls, and terminal states)
    for r in range(ROWS):
        row_output = ""
        for c in range(COLS):

            if (r, c) == agent_pos:
                row_output += " A "
            elif (r, c) in terminal_positions:
                row_output += f"{reward_map[r][c]:3d}"
            elif (r, c) in blocked_positions:
                row_output += " X "
            else:
                row_output += " . "
        print(row_output)
    print()


def print_q_table():
    # display all Q-values for every positio
    print("CURRENT Q-TABLE:\n")
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) in terminal_positions:
                print(f"{(r,c)} -> TERMINAL")
            elif (r, c) in blocked_positions:
                print(f"{(r,c)} -> BLOCKED")
            else:
                print(f"{(r,c)} -> {q_values[r][c]}")
        print()
    print("-" * 50)


# training loop (EPOCHS)

TOTAL_EPOCHS = 50

for epoch in range(1, TOTAL_EPOCHS + 1):

    agent_position = start_position
    step_number = 0

    while agent_position not in terminal_positions:

        clear_screen()
        print(f"EPOCH {epoch} | STEP {step_number}")
        draw_gridworld(agent_position)

        # Priprintnt Q-table every step
        print_q_table()

        action = choose_action(agent_position)
        new_q = updated_q_value(agent_position, action)
        set_q(agent_position, action, new_q)

        agent_position = next_position(agent_position, action)
        step_number += 1

        time.sleep(0.15)

    clear_screen()
    print(f"EPOCH {epoch} COMPLETE!")
    draw_gridworld(agent_position)
    print_q_table()

    time.sleep(1.0)
