import os

class td_qlearning:

  # possible actions; used for finding a* in policy()
  possible_actions = [1, 2, 3] # number of coins to take from bag
  state_actions_each_trial = []
  explored_state_action_pairs = []
  alpha = 0.10
  gamma = 0.90
  # define convergence
  threshold = 0.001 # minimum change needed to be observed to continue updates

  # store Q-function in class variable (a dictionary with key:(state, action) value:qvalue)
  qfunction = {}

  def __init__(self, directory):
    # directory is the path to a directory containing trials through state space

    # learn Q-function from CSV file(s)
    for root, dirs, files in os.walk(directory):
      for file in files:
        if file.endswith(".csv"):
          file_path = os.path.join(root, file)
          with open(file_path, 'r') as f:
            state_actions = []
            for line in f:
              # skip empty lines
              line = line.strip()
              if not line:
                continue

              # parse state and action from line
              state_str, value_str = line.split(',')
              # store state as variables in a tuple
              # (c_bag, c_agent, c_opponent, winner)
              s = tuple(state_str.split('/'))
              state = (int(s[0]), int(s[1]), int(s[2]), s[3])
              

              # convert action part to int, or keep as - for terminal state
              try:
                action = int(value_str)
              except ValueError:
                action = '-'
              state_actions.append((state, action))
            
            self.state_actions_each_trial.append(state_actions)
    
    self.explored_state_action_pairs = list(set([pair for sublist in self.state_actions_each_trial for pair in sublist]))
    
    # initilize Q(s,a)
    for trial in self.state_actions_each_trial:
      for (state, action) in trial:          
          self.qfunction[(state, action)] = self.reward(state)

    # update Q-value functions until convergence
    max_diff = float('inf')
    num_trials = len(self.state_actions_each_trial)
    while max_diff >= self.threshold:
        max_diff = 0
        for j in range(num_trials):
          state_actions_pairs = self.state_actions_each_trial[j]
          for k in range(len(state_actions_pairs) - 1):
            prev_state = state_actions_pairs[k][0]
            prev_action = state_actions_pairs[k][1]
            current_state = state_actions_pairs[k + 1][0]
            diff = self.temporal_difference_qlearning(prev_state, prev_action, current_state)
            max_diff = max(abs(diff), max_diff)

  def temporal_difference_qlearning(self, prev_state, prev_action, current_state):
    # consider possible actions
    action_qvalues = []
    for action in self.possible_actions:
      # if action is a terminal state, no action can be taken so use "-" action
      if (current_state[3] != '-'):
        action_qvalues.append(self.qvalue(current_state, "-"))
      # action is only valid if there are sufficient coins in the bag
      elif (current_state[0] - action) >= 0:
        action_qvalues.append(self.qvalue(current_state, action))
    
    # perform Q-value update
    if (action_qvalues):
      error_term = self.reward(prev_state) + self.gamma*max(action_qvalues) - self.qvalue(prev_state, prev_action)
      self.qfunction[prev_state, prev_action] += self.alpha*error_term

    return error_term

  def qvalue(self, state, action):
    # state is a string representation of a state
    # action is an integer representation of an action

    # find the qvalue of the given state action pair
    if ((state,action) in self.explored_state_action_pairs):
        return self.qfunction[state, action]
    else:
      # if not seen, qvalue defaults to would-be initial value reward(s)
      return self.reward(state)

  def policy(self, state):
    # state is a string representation of a state

    # terminal state, so no action can be taken
    if state[3] != '-':
      return 0

    # find the action that maximizes the qvalue
    best_action = self.possible_actions[0]
    for action in self.possible_actions[1:]:
      # only consider legal moves
      if (state[0] >= action):
        if self.qvalue(state, action) > self.qvalue(state, best_action):
          best_action = action

    # Return the optimal action (as an integer) under the learned policy
    return best_action
  
  def reward(self,state):
    # helper function to calculate the reward value of a state
    # assumes state is a tuple (c_bag, c_agent, c_opponent, winner)

    match state[3]:
      case "A":
        return int(state[1])
      case "O":
        return -int(state[1])
      case _:
        return 0
      
# test the class
if __name__ == "__main__":
    learner = td_qlearning("Examples/Example1/Trials")  # folder containing CSVs
    # print(len(td_qlearning.explored_state_action_pairs))
    
    # print all learned Q-values
    print("Learned Q-function:")
    for key, value in learner.qfunction.items():
        print(f"{key}: {value}")

    #Example: test a state and get the best action

    # test policy

    print("\nTest policy\n")

    # example0


    # example1
    print(learner.policy((6, 1, 6, '-')))
    print(learner.policy((0, 7, 6, 'O')))
    print(learner.policy((1, 8, 4, '-')))
    print(learner.policy((1, 6, 6, '-')))
    print(learner.policy((4, 5, 4, '-')))
    print(learner.policy((2, 8, 3, '-')))
    print(learner.policy((9, 2, 2, '-')))
    print(learner.policy((3, 6, 4, '-')))
    print(learner.policy((1, 5, 7, '-')))
    print(learner.policy((1, 9, 3, '-')))

    print("\nTest qvalue\n")

    # example0
    # print(learner.qvalue((8,3,2,'-'),2))

    # example1
    print(learner.qvalue((6, 1, 6, '-'), 2))
    print(learner.qvalue((0, 7, 6, 'O'), 0))
    print(learner.qvalue((1, 8, 4, '-'), 1))
    print(learner.qvalue((1, 6, 6, '-'), 1))
    print(learner.qvalue((4, 5, 4, '-'), 3))
    print(learner.qvalue((2, 8, 3, '-'), 2))
    print(learner.qvalue((9, 2, 2, '-'), 3))
    print(learner.qvalue((3, 6, 4, '-'), 2))
    print(learner.qvalue((1, 5, 7, '-'), 1))
    print(learner.qvalue((1, 9, 3, '-'), 1))
