import os

class MuJoCoSceneGenerator:
    def __init__(self, scene_config):
        self.config = scene_config

    #example config
    # {
    #             "attach_at_site": "payload_s",
    #             "model": "blocks/cf2.xml",
    #             "cable_length": 0.2,
    #             "cable_bodies": 20,
    #             "cable_mass": 0.005,
    #             "cable_damping": 0.00001,
    #             "cable_color_rgba": "0.1 0.1 0.8 1",
    #             "quad_attachment_site": "quad_attachment",
    #             "quad_attachment_offset": [0, 0, 0],
    #         }

    def generate_rope(self, quad_config, quad_id):
        # retrieve rope parameters from the config (with defaults)
        cable_config = quad_config["cable"]
        cable_length     = cable_config.get("length", 0.2)
        cable_bodies     = cable_config.get("bodies", 20)
        cable_mass_total = cable_config.get("mass", 0.005)
        cable_damping    = cable_config.get("damping", 0.00001)
        cable_color_rgba = cable_config.get("color_rgba", "0.1 0.1 0.8 1")

        # Compute per-segment values
        delta     = cable_length / cable_bodies          # offset for each nested body
        halfDelta = delta / 2                           # used for geom pos and capsule half-length
        mass_per_body = cable_mass_total / cable_bodies   # mass per geom

        # Start building the XML string
        cable_xml_lines = []
        # first body
        cable_xml_lines.append(f'<body name="q{quad_id}_cable_B_first">')
        cable_xml_lines.append(f'    <geom name="q{quad_id}_cable_G0" size="0.002 {halfDelta}" pos="{halfDelta} 0 0"')
        cable_xml_lines.append(f'        quat="0.707107 0 -0.707107 0" type="capsule" condim="1"')
        cable_xml_lines.append(f'        mass="{mass_per_body}" rgba="{cable_color_rgba}" />')
        cable_xml_lines.append(f'    <site name="q{quad_id}_cable_S_first" pos="0 0 0" group="3" />')
        cable_xml_lines.append(f'    <plugin instance="compositecable{quad_id}_" />')
        
        # intermediate bodies (if any)
        # For bodies 1 to cable_bodies-2, we nest them inside the previous body
        for i in range(1, cable_bodies - 1):
            cable_xml_lines.append(f'    <body name="q{quad_id}_cable_B_{i}" pos="{delta} 0 0">')
            cable_xml_lines.append(f'         <joint name="q{quad_id}_cable_J_{i}" pos="0 0 0" type="ball" group="3"')
            cable_xml_lines.append(f'             actuatorfrclimited="false" damping="{cable_damping}" />')
            cable_xml_lines.append(f'         <geom name="q{quad_id}_cable_G{i}" size="0.002 {halfDelta}" pos="{halfDelta} 0 0"')
            cable_xml_lines.append(f'             quat="0.707107 0 -0.707107 0" type="capsule" condim="1"')
            cable_xml_lines.append(f'             mass="{mass_per_body}" rgba="{cable_color_rgba}" />')
            cable_xml_lines.append(f'         <plugin instance="compositecable{quad_id}_" />')
        
        # last body: close with a site and plugin
        cable_xml_lines.append(f'    <body name="q{quad_id}_cable_B_last" pos="{delta} 0 0">')
        cable_xml_lines.append(f'         <joint name="q{quad_id}_cable_J_last" pos="0 0 0" type="ball" group="3"')
        cable_xml_lines.append(f'             actuatorfrclimited="false" damping="{cable_damping}" />')
        cable_xml_lines.append(f'         <geom name="q{quad_id}_cable_G_last" size="0.002 {halfDelta}" pos="{halfDelta} 0 0"')
        cable_xml_lines.append(f'             quat="0.707107 0 -0.707107 0" type="capsule" condim="1"')
        cable_xml_lines.append(f'             mass="{mass_per_body}" rgba="{cable_color_rgba}" />')
        cable_xml_lines.append(f'         <site name="q{quad_id}_cable_S_last" pos="{delta} 0 0" group="3" />')
        cable_xml_lines.append(f'         <plugin instance="compositecable{quad_id}_" />')
        
        rope = "\n".join(cable_xml_lines)

        cable_close = ""

        # close all open <body> tags.
        # We opened one tag for the first body, then one for every intermediate and one for the last body.
        # Total open bodies = cable_bodies (first + (cable_bodies-2) intermediates + last)
        for _ in range(cable_bodies):
            cable_close += "</body>\n"
        
        return rope, cable_close

    def generate_quad(self, quad_config):
        id = quad_config["id"]
        yaw_angle = quad_config.get("yaw_angle", 0)
        quad_header = f"""
            <!-- Quad {id} -->
            <body name="q{id}_cable_chain" pos="0 0 0.01" euler="0 0 {yaw_angle}">
            """
        quad = f"""
            <body
                name="q{id}_cf2"
                childclass="cf2"
                pos="0.0157895 0 0">
                <inertial
                    pos="0 0 0"
                    mass="0.034"
                    diaginertia="1.65717e-05 1.66556e-05 2.92617e-05" />
                <joint
                    type="ball"
                    pos="0 0 0"
                    limited="false"
                    damping="1e-05" />
                <geom
                    class="visual"
                    material="propeller_plastic"
                    mesh="cf2_0" />
                <geom
                    class="visual"
                    material="medium_gloss_plastic"
                    mesh="cf2_1" />
                <geom
                    class="visual"
                    material="polished_gold"
                    mesh="cf2_2" />
                <geom
                    class="visual"
                    material="polished_plastic"
                    mesh="cf2_3" />
                <geom
                    class="visual"
                    material="burnished_chrome"
                    mesh="cf2_4" />
                <geom
                    class="visual"
                    material="body_frame_plastic"
                    mesh="cf2_5" />
                <geom
                    class="visual"
                    material="white"
                    mesh="cf2_6" />
                <geom
                    class="collision"
                    mesh="cf2_collision_0" />
                <geom
                    class="collision"
                    mesh="cf2_collision_1" />
                <geom
                    class="collision"
                    mesh="cf2_collision_2" />
                <geom
                    class="collision"
                    mesh="cf2_collision_3" />
                <geom
                    class="collision"
                    mesh="cf2_collision_4" />
                <geom
                    class="collision"
                    mesh="cf2_collision_5" />
                <geom
                    class="collision"
                    mesh="cf2_collision_6" />
                <geom
                    class="collision"
                    mesh="cf2_collision_7" />
                <geom
                    class="collision"
                    mesh="cf2_collision_8" />
                <geom
                    class="collision"
                    mesh="cf2_collision_9" />
                <geom
                    class="collision"
                    mesh="cf2_collision_10" />
                <geom
                    class="collision"
                    mesh="cf2_collision_11" />
                <geom
                    class="collision"
                    mesh="cf2_collision_12" />
                <geom
                    class="collision"
                    mesh="cf2_collision_13" />
                <geom
                    class="collision"
                    mesh="cf2_collision_14" />
                <geom
                    class="collision"
                    mesh="cf2_collision_15" />
                <geom
                    class="collision"
                    mesh="cf2_collision_16" />
                <geom
                    class="collision"
                    mesh="cf2_collision_17" />
                <geom
                    class="collision"
                    mesh="cf2_collision_18" />
                <geom
                    class="collision"
                    mesh="cf2_collision_19" />
                <geom
                    class="collision"
                    mesh="cf2_collision_20" />
                <geom
                    class="collision"
                    mesh="cf2_collision_21" />
                <geom
                    class="collision"
                    mesh="cf2_collision_22" />
                <geom
                    class="collision"
                    mesh="cf2_collision_23" />
                <geom
                    class="collision"
                    mesh="cf2_collision_24" />
                <geom
                    class="collision"
                    mesh="cf2_collision_25" />
                <geom
                    class="collision"
                    mesh="cf2_collision_26" />
                <geom
                    class="collision"
                    mesh="cf2_collision_27" />
                <geom
                    class="collision"
                    mesh="cf2_collision_28" />
                <geom
                    class="collision"
                    mesh="cf2_collision_29" />
                <geom
                    class="collision"
                    mesh="cf2_collision_30" />
                <geom
                    class="collision"
                    mesh="cf2_collision_31" />
                <site
                    name="q{id}_imu"
                    pos="0 0 0" />
                <site
                    name="q{id}_thrust1"
                    pos="0.032527 -0.032527 0" />
                <site
                    name="q{id}_thrust2"
                    pos="-0.032527 -0.032527 0" />
                <site
                    name="q{id}_thrust3"
                    pos="-0.032527 0.032527 0" />
                <site
                    name="q{id}_thrust4"
                    pos="0.032527 0.032527 0" />
                <site name="q{id}_attachment" pos="0 0 0" />
                
            </body>          
                                                                                       
"""
        rope, cable_close = self.generate_rope(quad_config, id)
        return quad_header + rope + quad + cable_close + "</body>"
        

    def generate_xml(self):

        cable_instances = ""
        for quad in self.config["quads"]:
            cable_instances += f'<instance name="compositecable{quad["id"]}_" /> \n'

        header = f"""
    <mujoco model="CF2 scene">
    <compiler angle="radian" meshdir="assets/" />

    <option timestep="0.004" density="1.225" viscosity="1.8e-05" integrator="implicit" />

    <visual>
        <global azimuth="-20" elevation="-20" ellipsoidinertia="true" />
        <headlight ambient="0.3 0.3 0.3" diffuse="0.6 0.6 0.6" specular="0 0 0" />
        <rgba haze="0.15 0.25 0.35 1" />
    </visual>

    <statistic meansize="0.05" extent="0.2" center="0 0 0.1" />

    <default>
        <default class="cf2">
            <site group="5" />
            <default class="visual">
                <geom type="mesh" contype="0" conaffinity="0" group="2" />
            </default>
            <default class="collision">
                <geom type="mesh" group="3" />
            </default>
        </default>
    </default>

    <extension>
        <plugin plugin="mujoco.elasticity.cable">
            {cable_instances}

        </plugin>
    </extension>

    <custom>
        <text name="composite_cable_" data="cable_cable_" />
    </custom>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512"
            height="3072" />
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
            rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
        <material name="polished_plastic" rgba="0.631 0.659 0.678 1" />
        <material name="polished_gold" rgba="0.969 0.878 0.6 1" />
        <material name="medium_gloss_plastic" rgba="0.109 0.184 0 1" />
        <material name="propeller_plastic" rgba="0.792 0.82 0.933 1" />
        <material name="white" />
        <material name="body_frame_plastic" rgba="0.102 0.102 0.102 1" />
        <material name="burnished_chrome" rgba="0.898 0.898 0.898 1" />
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"
            reflectance="0.2" />
        <mesh name="cf2_0" file="cf2_0.obj" />
        <mesh name="cf2_1" file="cf2_1.obj" />
        <mesh name="cf2_2" file="cf2_2.obj" />
        <mesh name="cf2_3" file="cf2_3.obj" />
        <mesh name="cf2_4" file="cf2_4.obj" />
        <mesh name="cf2_5" file="cf2_5.obj" />
        <mesh name="cf2_6" file="cf2_6.obj" />
        <mesh name="cf2_collision_0" file="cf2_collision_0.obj" />
        <mesh name="cf2_collision_1" file="cf2_collision_1.obj" />
        <mesh name="cf2_collision_2" file="cf2_collision_2.obj" />
        <mesh name="cf2_collision_3" file="cf2_collision_3.obj" />
        <mesh name="cf2_collision_4" file="cf2_collision_4.obj" />
        <mesh name="cf2_collision_5" file="cf2_collision_5.obj" />
        <mesh name="cf2_collision_6" file="cf2_collision_6.obj" />
        <mesh name="cf2_collision_7" file="cf2_collision_7.obj" />
        <mesh name="cf2_collision_8" file="cf2_collision_8.obj" />
        <mesh name="cf2_collision_9" file="cf2_collision_9.obj" />
        <mesh name="cf2_collision_10" file="cf2_collision_10.obj" />
        <mesh name="cf2_collision_11" file="cf2_collision_11.obj" />
        <mesh name="cf2_collision_12" file="cf2_collision_12.obj" />
        <mesh name="cf2_collision_13" file="cf2_collision_13.obj" />
        <mesh name="cf2_collision_14" file="cf2_collision_14.obj" />
        <mesh name="cf2_collision_15" file="cf2_collision_15.obj" />
        <mesh name="cf2_collision_16" file="cf2_collision_16.obj" />
        <mesh name="cf2_collision_17" file="cf2_collision_17.obj" />
        <mesh name="cf2_collision_18" file="cf2_collision_18.obj" />
        <mesh name="cf2_collision_19" file="cf2_collision_19.obj" />
        <mesh name="cf2_collision_20" file="cf2_collision_20.obj" />
        <mesh name="cf2_collision_21" file="cf2_collision_21.obj" />
        <mesh name="cf2_collision_22" file="cf2_collision_22.obj" />
        <mesh name="cf2_collision_23" file="cf2_collision_23.obj" />
        <mesh name="cf2_collision_24" file="cf2_collision_24.obj" />
        <mesh name="cf2_collision_25" file="cf2_collision_25.obj" />
        <mesh name="cf2_collision_26" file="cf2_collision_26.obj" />
        <mesh name="cf2_collision_27" file="cf2_collision_27.obj" />
        <mesh name="cf2_collision_28" file="cf2_collision_28.obj" />
        <mesh name="cf2_collision_29" file="cf2_collision_29.obj" />
        <mesh name="cf2_collision_30" file="cf2_collision_30.obj" />
        <mesh name="cf2_collision_31" file="cf2_collision_31.obj" />
    </asset>

    <worldbody>
        <geom name="goal_marker" size="0.02" pos="0 0 1" contype="0" conaffinity="0"
            rgba="1 0 0 0.8" />
        <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" />
        <light pos="0 0 1.5" dir="0 0 -1" directional="true" />

        <!-- Quad start site -->
        <site name="q1_start" pos="{self.config['quads'][0]['start_pos'][0]} {self.config['quads'][0]['start_pos'][1]} {self.config['quads'][0]['start_pos'][2]}" euler="{self.config['quads'][0]['start_euler'][0]} {self.config['quads'][0]['start_euler'][1]} {self.config['quads'][0]['start_euler'][2]}" />
        <site name="q0_start" pos="{self.config['quads'][1]['start_pos'][0]} {self.config['quads'][1]['start_pos'][1]} {self.config['quads'][1]['start_pos'][2]}" euler="{self.config['quads'][1]['start_euler'][0]} {self.config['quads'][1]['start_euler'][1]} {self.config['quads'][1]['start_euler'][2]}" />
   
        <!-- Payload start site-->
        <site name="payload_start" pos="{self.config['payload']['start_pos'][0]} {self.config['payload']['start_pos'][1]} {self.config['payload']['start_pos'][2]}" euler="{self.config['payload']['start_euler'][0]} {self.config['payload']['start_euler'][1]} {self.config['payload']['start_euler'][2]}" />




        <body name="payload" pos="0 0 0.1">
            <camera name="track" pos="-1 0 0.5" quat="0.601501 0.371748 -0.371748 -0.601501"
                mode="trackcom" />
            <joint type="free" actuatorfrclimited="false" />
            <geom size="0.007 0.01" type="cylinder" mass="0.001" rgba="0.8 0.8 0.8 1" />
            <site name="payload_s" pos="0 0 0.01" />
"""

        quads= ""
        for (i, quad) in enumerate(self.config["quads"]):
            q = quad
            pi = 3.14159
            yaw_angle = (2*pi / len(self.config["quads"])) * i - pi
            q["yaw_angle"] = yaw_angle
            quads += self.generate_quad(q)
            
        end = """
        </body>

        

    </worldbody>

    <contact>
        <exclude body1="q1_cable_B_first" body2="q1_cable_B_1" />
        <exclude body1="q1_cable_B_1" body2="q1_cable_B_2" />
        <exclude body1="q1_cable_B_2" body2="q1_cable_B_3" />
        <exclude body1="q1_cable_B_3" body2="q1_cable_B_4" />
        <exclude body1="q1_cable_B_4" body2="q1_cable_B_5" />
        <exclude body1="q1_cable_B_5" body2="q1_cable_B_6" />
        <exclude body1="q1_cable_B_6" body2="q1_cable_B_7" />
        <exclude body1="q1_cable_B_7" body2="q1_cable_B_8" />
        <exclude body1="q1_cable_B_8" body2="q1_cable_B_9" />
        <exclude body1="q1_cable_B_9" body2="q1_cable_B_10" />
        <exclude body1="q1_cable_B_10" body2="q1_cable_B_11" />
        <exclude body1="q1_cable_B_11" body2="q1_cable_B_12" />
        <exclude body1="q1_cable_B_12" body2="q1_cable_B_13" />
        <exclude body1="q1_cable_B_13" body2="q1_cable_B_14" />
        <exclude body1="q1_cable_B_14" body2="q1_cable_B_15" />
        <exclude body1="q1_cable_B_15" body2="q1_cable_B_16" />
        <exclude body1="q1_cable_B_16" body2="q1_cable_B_17" />
        <exclude body1="q1_cable_B_17" body2="q1_cable_B_last" />
        <exclude body1="q1_cable_B_last" body2="payload" />
        <exclude body1="q1_cable_B_last" body2="payload" />


     

        <exclude body1="q0_cable_B_first" body2="q0_cable_B_1" />
        <exclude body1="q0_cable_B_1" body2="q0_cable_B_2" />
        <exclude body1="q0_cable_B_2" body2="q0_cable_B_3" />
        <exclude body1="q0_cable_B_3" body2="q0_cable_B_4" />
        <exclude body1="q0_cable_B_4" body2="q0_cable_B_5" />
        <exclude body1="q0_cable_B_5" body2="q0_cable_B_6" />
        <exclude body1="q0_cable_B_6" body2="q0_cable_B_7" />
        <exclude body1="q0_cable_B_7" body2="q0_cable_B_8" />
        <exclude body1="q0_cable_B_8" body2="q0_cable_B_9" />
        <exclude body1="q0_cable_B_9" body2="q0_cable_B_10" />
        <exclude body1="q0_cable_B_10" body2="q0_cable_B_11" />
        <exclude body1="q0_cable_B_11" body2="q0_cable_B_12" />
        <exclude body1="q0_cable_B_12" body2="q0_cable_B_13" />
        <exclude body1="q0_cable_B_13" body2="q0_cable_B_14" />
        <exclude body1="q0_cable_B_14" body2="q0_cable_B_15" />
        <exclude body1="q0_cable_B_15" body2="q0_cable_B_16" />
        <exclude body1="q0_cable_B_16" body2="q0_cable_B_17" />
        <exclude body1="q0_cable_B_17" body2="q0_cable_B_last" />
        <exclude body1="q0_cable_B_last" body2="payload" />
        <exclude body1="q0_cable_B_last" body2="payload" />
    </contact>

    <!-- <equality>
    <connect site1="cable_S_last" site2="payload_s"/>
  </equality> -->

  <!-- quad start site -->
    <equality>
        <weld site1="q1_start" site2="q1_imu" solref="0.01 4"  />
        <weld site1="q0_start" site2="q0_imu" solref="0.01 4"  />
      
        <!-- 
        <weld site1="payload_start" site2="payload_s" solref="0.01 4"  />
        -->

    </equality>

   

    <actuator>
        <general name="q1_thrust1" class="cf2" site="q1_thrust1" ctrlrange="0 0.14"
            gear="0 0 1 0 0 6e-06" />
        <general name="q1_thrust2" class="cf2" site="q1_thrust2" ctrlrange="0 0.14"
            gear="0 0 1 0 0 -6e-06" />
        <general name="q1_thrust3" class="cf2" site="q1_thrust3" ctrlrange="0 0.14"
            gear="0 0 1 0 0 6e-06" />
        <general name="q1_thrust4" class="cf2" site="q1_thrust4" ctrlrange="0 0.14"
            gear="0 0 1 0 0 -6e-06" />

    

        <general name="q0_thrust1" class="cf2" site="q0_thrust1" ctrlrange="0 0.14"
            gear="0 0 1 0 0 6e-06" />
        <general name="q0_thrust2" class="cf2" site="q0_thrust2" ctrlrange="0 0.14"
            gear="0 0 1 0 0 -6e-06" />
        <general name="q0_thrust3" class="cf2" site="q0_thrust3" ctrlrange="0 0.14"
            gear="0 0 1 0 0 6e-06" />
        <general name="q0_thrust4" class="cf2" site="q0_thrust4" ctrlrange="0 0.14"
            gear="0 0 1 0 0 -6e-06" />
    </actuator>

    <sensor>
        <gyro site="q1_imu" name="body_gyro" />
        <accelerometer site="q1_imu" name="body_linacc" />
        <framequat objtype="site" objname="q1_imu" name="body_quat" />

        <!-- force at cable attachment site -->

        <force name="q0_cable_force" site="q0_attachment" />
        <force name="q1_cable_force" site="q1_attachment" />
  

    </sensor>
</mujoco>"""
        return header + quads + end

if __name__ == "__main__":
    scene_config = {
        "payload": {
            "mass": 0.01,
            "geom_type": "cylinder",
            "size": [0.007, 0.01],
            "start_pos": [0, 0, 0.2],
            "start_euler": [0, 1, 1],
            "color_rgba": "0.8 0.8 0.8 1",
            "attach_sites": [
                {
                    "name": "attach_site_1",
                    "pos": [0, 0, 0.01]
                },
                {
                    "name": "attach_site_2",
                    "pos": [0, 0, -0.01]
                }
            ]
        },
        "quad_prefix": "q",
        "quads": [
            {
                "id": 0,
                "start_pos": [0.15, 0, 0.3],
                "start_euler": [0, 1, 0],
                "cable":{
                    "length": 0.5,
                    "bodies": 25,
                    "mass": 0.01,
                    "damping": 0.00001,
                    "color_rgba" : "0.1 0.8 0.1 1",
                    "quad_site": "q1_attachment",
                    "attachment_offset": [0, 0.001, 0],
                    "payload_site": "attach_site_1",
                }
                
            },
            {
               
                "id": 1,
                "start_pos": [-0.15, 0, 0.3],
                "start_euler": [0, 0, 0.5],
                "cable":{
                    "length": 0.2,
                    "bodies": 20,
                    "mass": 0.005,
                    "damping": 0.00001,
                    "color_rgba" : "0.1 0.1 0.8 1",
                    "quad_site": "quad_attachment",
                    "attachment_offset": [0, 0.001, 0],
                    "payload_site": "attach_site_1",
                }
            },
         
        ]
    }
    generator = MuJoCoSceneGenerator(scene_config)
    full_xml = generator.generate_xml()
    output_file = os.path.join(os.path.dirname(__file__), "full_test.xml")
    with open(output_file, "w") as f:
        f.write(full_xml)
    print(f"Full mujoco xml saved to {output_file}")