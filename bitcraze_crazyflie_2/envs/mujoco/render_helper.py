
def render_rollout(env, rollout, camera="side"):
    # Implement a simple renderer if one isn’t defined.
    frames = []
    for state in rollout:
        # Use a dummy call; replace with your actual rendering code.
        frame = env.sys.render(camera=camera) if hasattr(env.sys, "render") else None
        if frame is not None:
            frames.append(frame)
    return frames
