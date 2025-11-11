import os

class td_qlearning:

  # possible actions; used for finding a* in policy()
  possible_actions = [1, 2, 3]

  alpha = 0.10
  gamma = 0.90

  # store Q-function in class variable (a dictionary with key:(state, action) value:qvalue)
  qfunction = {}

  def __init__(self, directory):
    # directory is the path to a directory containing trials through state space

    # NTS: may be better to store all (state, action) pairs in a list first (TODO 1/2)

    # learn Q-function from CSV file(s)
    for root, dirs, files in os.walk(directory):
      for file in files:
        if file.endswith(".csv"):
          file_path = os.path.join(root, file)
          with open(file_path, 'r') as f:
            for line in f:
              # skip empty lines
              line = line.strip()
              if not line:
                continue

              # parse state and action from line
              state_str, value_str = line.split(',')
              # store state as variables in a tuple
              # (c_bag, c_agent, c_opponent, winner)
              state = tuple(state_str.split('/'))

              # convert action part to int, or keep as - for terminal state
              try:
                action = int(value_str)
              except ValueError:
                action = value_str

              # compute qvalue for qfunction (TODO 2/2)

  def qvalue(self, state, action):
    # state is a string representation of a state
    # action is an integer representation of an action

    # fund the qvalue of the given state action pair
    qvalue = self.qfunction[state, action]

    # Return the q-value for the state-action pair
    return qvalue

  def policy(self, state):
    # state is a string representation of a state

    # find the action that maximizes the qvalue
    best_action = 1
    for action in self.possible_actions:
      qvalue = self.qfunction[state, action]
      if qvalue > self.qfunction[state, best_action]:
        best_action = action

    # Return the optimal action (as an integer) under the learned policy
    return best_action
  
  def reward(state):
    # helper function to calculate the reward value of a state
    # assumes state is a tuple (c_bag, c_agent, c_opponent, winner)

    match state[3]:
      case "A":
        return state[1]
      case "O":
        return -state[1]
      case _:
        return 0
      
# test the class
if __name__ == "__main__":
    learner = td_qlearning("Examples/Example0/Trials")  # folder containing CSVs

    # print all learned Q-values
    print("Learned Q-function:")
    for key, value in learner.qfunction.items():
        print(f"{key}: {value}")

    # Example: test a state and get the best action
    example_state = ('13', '0', '0', '-')
    best_action = learner.policy(example_state)
    print(f"\nBest action for state {example_state}: {best_action}")