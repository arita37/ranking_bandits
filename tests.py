"""
Test script for the save and load functionality of the TOP_RANK class.

   export pyinstrument=1  ### enable pyinstrument tracing

   python tests.py test_toprank



This script creates an instance of the TOP_RANK class, simulates a game by performing actions,
saves the model using the save_model function, and then loads the model parameters into a new instance
using the load_params function. It checks whether the loaded parameters match the original player's parameters.

The purpose of this script is to test the save and load functionality of the TOP_RANK class to ensure
that model persistence works as expected.

Usage:
    Run this script to perform the save and load test for the TOP_RANK class.

"""

import numpy as np
from bandits_to_rank.opponents.top_rank import TOP_RANK  
import pyinstrument

from utilmy import (log, os_makedirs)


##############################################################################################
def test_toprank():
    # Specify the number of arms, positions, and discount factors
    nb_arms = 10
    discount_factors = [0.9, 0.8, 0.7]  # A list of discount factors for each position
    T = 1000  # Provide a value for 'T'

    # Create an instance of your TOP_RANK class
    player = TOP_RANK(nb_arms, T=T, discount_factor=discount_factors)

    # Perform some updates or actions to simulate a game
    # This is just a dummy example; you should replace it with actual game actions
    for _ in range(100):
        choices, _ = player.choose_next_arm()  # Capture the choices and ignore the second value
        rewards = np.where(np.arange(nb_arms) == choices[0], 1, 0)  # Use NumPy's element-wise comparison
        player.update(choices, rewards)

    # Save the model
    player.save_model(dirout="models")

    # Create a new instance of the TOP_RANK class
    new_player = TOP_RANK(nb_arms, T=T, discount_factor=discount_factors)

    # Load the model parameters into the new instance
    new_player.load_params(dirout="models")

    # Verify that the loaded parameters match the original ones
    # You can add assertions or checks here to ensure consistency
    assert player.nb_arms == new_player.nb_arms
    assert player.known_discount == new_player.known_discount
    assert np.array_equal(player.discount_factor, new_player.discount_factor)
    assert player.time == new_player.time
    assert player.T_horizon == new_player.T_horizon
    assert player.graph == new_player.graph
    assert player.partition == new_player.partition
    assert np.array_equal(player.s, new_player.s)
    assert np.array_equal(player.n, new_player.n)

    log("Save and Load Test: Passed")


##############################################################################################
if __name__ == "__main__":
    if os.environ.get('pyinstrument', "0") == "1" :
       profiler = pyinstrument.Profiler()
       profiler.start()

       fire.Fire() 
       profiler.stop()
       print(profiler.output_text(unicode=True, color=True))
    else :
        fire.Fire()


