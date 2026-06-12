from rail_env_theia import TheiaCFEnv

env = TheiaCFEnv(difficulty_level=2, mode="antrenare", render_mode="human")

obs, info = env.reset(seed=2)
print("OBS shape:", obs.shape)
print("INFO reset:", info)

for action in [0, 1, 2, 3, 4, 5]:
    obs, info = env.reset(seed=2)

    print("\n" + "=" * 60)
    print(f"TEST actiune = {action}")

    obs2, reward, terminated, truncated, step_info = env.step(action)

    print("reward     =", reward)
    print("terminated =", terminated)
    print("truncated  =", truncated)
    print("step_info  =", step_info)