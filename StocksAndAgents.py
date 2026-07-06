import ImpactFunctions

import numpy as np


# The goal of this file is to create a general process, and then compute the costs from the trading of the agents

class AuxilliaryStockProcesses:
    # To compute cost and impact processes for an individual stock
    def __init__(self, times, agents, impact_functions, trading_cost_functions, stock_name, stock_value, stock_series = None):
        self.times = times # np array of times, starting at 0
        self.agents = agents # np array or list of agents
        self.impact_function = impact_functions[stock_name] # ImpactProcesses - to compute impacts, costs due to impact 
        self.trading_cost_function = trading_cost_functions[stock_name]  # TradingCost - to compute trading costs, costs due to trades
        self.stock_name = stock_name # Stock name
        self.stock_value = stock_value # Stock initial value
        self.stock_series = stock_series # Stock unperturbed values

        self.velocity = None
        self.volume = None
        self.impact_costs = None
        self.impact = None
        self.trading_costs = None
        self.unperturbed_liquidation_values = None
        self.perturbed_liquidation_values = None


    def compute_impact_costs(self):
        """Computes the impact costs for each agent for the stock"""
        trading_processes = {agent.name: agent.trading_processes[self.stock_name] for agent in self.agents} # Dictionary of trading processes
        agent_names = [agent.name for agent in self.agents] # List of names (keys)

        cost_proccesses = dict()
        for name_i in agent_names: # trader
            costs_array = 0
            for name_j in agent_names: # impact costs trading against
                impact_process = trading_processes[name_j]
                trading_process = trading_processes[name_i]
                costs_array += self.impact_function.impact_cost(impact_process, trading_process)
            cost_proccesses[name_i] = costs_array * self.stock_value

        self.impact_costs = cost_proccesses 

        return self.impact_costs 
        
    def compute_impact(self):
        """Computes the impact for the stock"""
        trading_processes = {agent.name: agent.trading_processes[self.stock_name] for agent in self.agents} # Dictionary of trading processes
        agent_names = [agent.name for agent in self.agents] # List of names (keys)

        impact = 0

        for name in agent_names:
            impact += self.impact_function.compute_impact(trading_processes[name])

        self.impact = impact * self.stock_value

        return self.impact
        
    def compute_trading_costs(self):
        """Computes the trading costs for each agent for the stock"""
        trading_processes = {agent.name: agent.trading_processes[self.stock_name] for agent in self.agents} # Dictionary of trading processes
        agent_names = [agent.name for agent in self.agents] # List of names (keys)

        cost_proccesses = dict()
        for name_i in agent_names: # trader
            costs_array = 0
            trading_process_i = trading_processes[name_i]
            for name_j in agent_names: # impact costs trading against
                trading_process_j = trading_processes[name_j]            
                costs = self.trading_cost_function.compute_costs(trading_process_i, trading_process_j)
                costs_array += costs
            cost_proccesses[name_i] = costs_array * self.stock_value

        self.trading_costs = cost_proccesses 

        return self.trading_costs
    
    def trading_velocity(self):
        """Computes the trading velocity of the agents"""
        velocity = 0

        for agent in self.agents:
            velocity += agent.trading_processes[self.stock_name].velocity

        self.velocity = velocity

        return self.velocity
    
    def trading_volume(self):
        """Computes the trading volume of the agents"""
        volume = 0

        for agent in self.agents:
            volume += agent.trading_processes[self.stock_name].volume

        self.volume = volume

        return self.volume





class Agent:
    # Portfolio and liquidation strategy for an agent
    def __init__(self, trading_processes_dict, name = 'none'):
        self.trading_processes = trading_processes_dict # dictionary of trading processes, organised by stock name
        self.name = name # Set the name of the agent
