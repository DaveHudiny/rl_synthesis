# PAYNT + DRL

This toolkit implements the PAYNT-DRL loop. The implementation contains the modified version of the PAYNT project and our DRL extension containing SOTA methods for POMDP planning using LSTM networks.



## Installation
  The installation was tried on PC with Ubuntu 24.04. Before the installation, please install python3.10 (https://askubuntu.com/questions/682869/how-do-i-install-a-different-python-version-using-apt-get), since the official version of TensorFlow Agents currently does not support newer versions of Python. 
  Then run ./install.sh script. Or build a new docker image using our integrated Dockerfile. Furthermore, if you create your own environemnt, you can install through pip install . using the pyproject.toml file.

## Usage and the Expected Output

If you want to run RL for purely single POMDP setting (without the family of size 1), run the simple_rl.py script.

If you want to experiment with our robust RL loop, please use the robust_pomdps_rl.py script.

## Structure
  - models -- set of all models we tried to experiment with.
  - VecStorm -- implementation of our vectorized simulator.
  - rl_src -- main implementations of the reinforcement learning part.
  - paynt -- extended PAYNT implementation.
  - robust_rl -- core parts of our robust training loop.

## Framework and Sources
The implementation is primarily based on PAYNT with Stormpy, TensorFlow Agents framework, which implements many important blocks of this project as reinforcement learning algorithms, and the AALpy Automata learning library (https://github.com/DES-Lab/AALpy). We also took some inspiration and in case of .rl_src/environment/pomdp_builder, we took the code from repository: [Shielding](https://github.com/stevencarrau/safe_RL_POMDPs).

## Credits

- Main author: David Hudák (ihudak@fit.vutbr.cz)
- PAYNT: Roman Andriushchenko (https://github.com/randriu) and Filip Macák (https://github.com/TheGreatfpmK)
- VecStorm: Modified version of the implementation by Martin Kurečka (https://github.com/kurecka/VecStorm)
- Automata learning implementation: Martin Tappler (https://github.com/mtappler)
- Technical and theoretical consultations: Maris F.L. Galesloot (maris.galesloot@ru.nl)
- Supervisor: Milan Češka (ceskam@fit.vutbr.cz)
