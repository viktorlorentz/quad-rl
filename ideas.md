* Jerk in observation and reward
    * Addition to obs needed or stacking enough?

* Potentially use Jax / Mujoco MJX

* Potentially pretrain with IL
    * Use expert to pretrain policy with IL before doing RL

* Disturbance Correction
    *Apply Random Forces before attaching payloads to encourage disturbance correction learning

* Ideas from https://arxiv.org/pdf/2403.12203
    * Train Critic with priveleged info maybe freeze actor
    * Dynamically change Clip range
    * Assymetric  Priveleged Critic
    * Spherical coords for OBS pos error and payload error

* Set starting pos as origin frame instead of fully local



