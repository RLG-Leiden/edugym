def play_env(env, input_prompt, input_to_action):
    state = env.reset()

    # Take some random actions in the environment
    env.render()

    done = False
    while not done:
        action_input = input(
            f"Provide an action. {input_prompt}.\nAny other key will exit execution \n"
        )
        if action_input in input_to_action:
          action = input_to_action[action_input]
        else:
          break

        next_state, reward, done, truncated, info = env.step(action)

        # Render the environment
        env.render()

        print(
            f"State: {state}, Action: {action}, Next state {next_state}, Reward: {reward}, Done: {done}"
        )

        if done:
            state = env.reset()
        else:
            state = next_state

    # Close the environment
    env.close()