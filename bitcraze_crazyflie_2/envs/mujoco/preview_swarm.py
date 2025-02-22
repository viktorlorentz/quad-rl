import os  # new
import time
import mujoco
import mujoco.viewer
from generate_swarm import QuadSceneGenerator  # ...existing code...

def main():
    # Load assets from the "assets" folder located with the script
    asset_dir = os.path.join(os.path.dirname(__file__), "assets")
    assets = {}
    for filename in os.listdir(asset_dir):
        file_path = os.path.join(asset_dir, filename)
        with open(file_path, "rb") as f:
            assets[filename] = f.read()

    generator = QuadSceneGenerator({})
    xml = generator.generate_random()  # generate random swarm XML
    # Change: pass the assets dict, not asset_dir
    m = mujoco.MjModel.from_xml_string(xml, assets=assets)
    d = mujoco.MjData(m)
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while True:
            xml = generator.generate_random()  # generate random swarm XML
            # Change: pass the assets dict, not asset_dir
            model = mujoco.MjModel.from_xml_string(xml, assets=assets)
            data = mujoco.MjData(model)
            with viewer.lock():
                viewer.mjData = data
                viewer.mjModel = model
            viewer.sync()
            start_time = time.time()
            while viewer.is_running() and time.time() - start_time < 2:
                step_start = time.time()
                mujoco.mj_step(model, data, nstep=1)
           
                time_until_next = model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)

if __name__ == "__main__":
    main()
