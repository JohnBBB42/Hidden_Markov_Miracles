class FHMM:
    def __init__(self, num_factors, state_dim, transition_probs=None, emission_probs=None):
        # Initialize FHMM with given number of factors and discrete state space size.
        self.num_factors = num_factors
        self.state_dim = state_dim
        # If no transition matrix given, use identity (no change) for each factor as default.
        if transition_probs is None:
            trans = torch.eye(state_dim)
            # Each factor has its own transition matrix; assume identical for simplicity
            self.transition_probs = torch.stack([trans for _ in range(num_factors)])
        else:
            self.transition_probs = transition_probs
        # Emission probabilities (for discrete outputs) or a function for continuous outputs.
        self.emission_probs = emission_probs
    
    def transition_prob(self, prev_states, current_states):
        """Compute the probability of transitioning from prev_states to current_states."""
        prob = 1.0
        for f in range(self.num_factors):
            ps, cs = prev_states[f], current_states[f]
            prob *= self.transition_probs[f, ps, cs].item()
        return prob
    
    def emission_prob(self, states, observation):
        """Probability of an observation given the hidden states."""
        if self.emission_probs is not None:
            if isinstance(observation, int):  # Categorical observation
                prob = 1.0
                for f in range(self.num_factors):
                    prob *= self.emission_probs[states[f], observation]
                return prob
            else:
                # For continuous observations, define a likelihood function as needed.
                raise NotImplementedError("Continuous emission probability not implemented.")
        else:
            # If not defined, assume the neural network handles emissions.
            return 1.0
