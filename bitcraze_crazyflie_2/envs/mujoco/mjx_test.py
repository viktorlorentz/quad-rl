import os
import time
import numpy as np
import mujoco
import jax
import mujoco.mjx as mjx
import matplotlib.pyplot as plt


jax.config.update("jax_compilation_cache_dir",  "/tmp/jax-cache")  # current directory
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


# Find and load the XML file
xml_path = os.path.join(os.path.dirname(__file__), "two_quad_payload.xml")
with open(xml_path, "r") as f:
    xml = f.read()

# Make mujoco model, data, and renderer
mj_model = mujoco.MjModel.from_xml_string(xml)
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model)

# Convert to MJX model and data
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

# Set up visualization options
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

# Simulation parameters
duration = 5.0  # seconds
framerate = 30  # Hz

# Helper function to save videos
def save_video(frames, filename, fps=30):
    try:
        import cv2
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        for frame in frames:
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video.release()
        print(f"Video saved to {filename}")
    except ImportError:
        try:
            import imageio
            imageio.mimsave(filename, frames, fps=fps)
            print(f"Video saved to {filename}")
        except ImportError:
            print("Could not save video. Install OpenCV or imageio.")
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(frames[0])
            plt.title("First Frame")
            plt.subplot(1, 2, 2)
            plt.imshow(frames[-1])
            plt.title("Last Frame")
            plt.savefig(f"{filename.split('.')[0]}_frames.png")

# Part 1: Regular simulation and video rendering
print("Running regular MuJoCo simulation...")
frames = []
mujoco.mj_resetData(mj_model, mj_data)

start_time = time.time()
while mj_data.time < duration:
    mujoco.mj_step(mj_model, mj_data)
    if len(frames) < mj_data.time * framerate:
        renderer.update_scene(mj_data, scene_option=scene_option)
        pixels = renderer.render()
        frames.append(pixels)
mujoco_time = time.time() - start_time

print(f"MuJoCo simulation time: {mujoco_time:.4f} seconds")
save_video(frames, "mujoco_sim.mp4", framerate)

# Part 2: MJX simulation
print("Running MJX simulation...")
frames = []
mujoco.mj_resetData(mj_model, mj_data)
mjx_data = mjx.put_data(mj_model, mj_data)

# JIT-compile the step function
jit_step = jax.jit(mjx.step)

start_time = time.time()
while mjx_data.time < duration:
    mjx_data = jit_step(mjx_model, mjx_data)
    if len(frames) < mjx_data.time * framerate:
        mj_data = mjx.get_data(mj_model, mjx_data)
        renderer.update_scene(mj_data, scene_option=scene_option)
        pixels = renderer.render()
        frames.append(pixels)
mjx_single_time = time.time() - start_time

print(f"MJX single simulation time: {mjx_single_time:.4f} seconds")
save_video(frames, "mjx_sim.mp4", framerate)

# Part 3: Batched simulation with 512 environments and 10000 timesteps
batch_size = 64
num_steps = 10000
print(f"Running batched MJX simulation with {batch_size} environments for {num_steps} steps...")

# Create a batch of initial states with small random variations
rng = jax.random.PRNGKey(0)
keys = jax.random.split(rng, batch_size)

# Initialize batch with small random variations
def init_batch_fn(key):
    return mjx_data.replace(
        qpos=mjx_data.qpos + jax.random.uniform(key, mjx_data.qpos.shape, minval=-0.01, maxval=0.01)
    )

batch = jax.vmap(init_batch_fn)(keys)

# Compile the batched step function
jit_batch_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

# Run the batched simulation
start_time = time.time()
for _ in range(num_steps):
    batch = jit_batch_step(mjx_model, batch)
batch_sim_time = time.time() - start_time

print(f"MJX batched simulation time: {batch_sim_time:.4f} seconds")
print(f"Average time per environment step: {batch_sim_time / (batch_size * num_steps) * 1e6:.2f} microseconds")
print(f"Speedup over MuJoCo (normalized): {(mujoco_time / duration) / (batch_sim_time / (batch_size * num_steps)):.2f}x")

# Summary
print("\nPerformance Summary:")
print("-" * 50)
print(f"MuJoCo Simulation Time: {mujoco_time:.4f} s")
print(f"MJX Single Simulation Time: {mjx_single_time:.4f} s")
print(f"MJX Batched Simulation Time ({batch_size} x {num_steps}): {batch_sim_time:.4f} s")
print(f"MJX Batched Time per Step: {batch_sim_time / num_steps:.6f} s")
print(f"MJX Batched Time per Environment Step: {batch_sim_time / (batch_size * num_steps) * 1e6:.2f} µs")

# Save the performance results
results = {
    "mujoco_time": mujoco_time,
    "mjx_single_time": mjx_single_time,
    "batch_size": batch_size,
    "num_steps": num_steps,
    "batch_sim_time": batch_sim_time,
    "time_per_env_step_us": batch_sim_time / (batch_size * num_steps) * 1e6,
    "speedup": (mujoco_time / duration) / (batch_sim_time / (batch_size * num_steps))
}

# Create performance visualization
labels = ['MuJoCo', 'MJX Single', f'MJX Batched\n(per env)']
times = [
    mujoco_time / (duration / mj_model.opt.timestep), 
    mjx_single_time / (duration / mj_model.opt.timestep),
    batch_sim_time / batch_size / num_steps * 1e6  # microseconds per step
]

plt.figure(figsize=(10, 6))
plt.bar(labels, times, color=['blue', 'green', 'red'])
plt.yscale('log')
plt.ylabel('Time per step (log scale)')
plt.title('Simulation Performance Comparison')
plt.savefig('performance_comparison.png')
print("Performance visualization saved to performance_comparison.png")