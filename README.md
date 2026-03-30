### 1. Clone the repository:
   ```bash
   git clone https://github.com/mxmmbd/Safe-Path-Generation--Franka-Panda-Arm.git
   cd Safe-Path-Generation--Franka-Panda-Arm
   ```

### 2. Create a virtual environment(Optional but recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

### 3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### 4. Control the robot with Jacobian-based velocity control

   ```bash
   python3 Franka_with_Obstacle.py
   ```
   -> 3 obstacles(box-shaped), one goal point(green sphere) 

### 5. Also implemented a viapoint logic

   ```bash
   python3 Franka_Viapoint.py
   ```
