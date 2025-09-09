# PillagerBench: Benchmarking LLM-Based Agents in Competitive Minecraft Team Environments

We aim to investigate multi-agent systems in competitive team-vs-team scenarios in the Minecraft environment, and explore effective reinforcement learning techniques that enhance tactical play of Large Language Models.

Welcome to PillagerBench, where the blocky world of Minecraft isn't just for fun and games, it's a warzone for the machines! Our benchmark suite is designed to push the boundaries of what virtual agents can achieve by adding dimensions of competition and team play to create complex and dynamic state spaces.

Customize your benchmark with our PillagerAgent extensible API, allowing you to add custom scenarios and new multi-agent systems.
<p align="center">
    <a href='https://arxiv.org/pdf/2509.06235'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=arXiv&logoColor=green' alt='Paper PDF'>
    </a>
</p>

---

## Setup and Configuration

### Requirements
- **API Keys**: Obtain API keys from one or more of the following services:
  - OpenAI (for access to models like GPT-4o)
  - DeepSeek (for access to DeepSeek models)
  - OpenRouter (for access to a wide range of models)
- **Ollama (optional)**: You have the option to use models from Ollama running locally.

### Docker Installation Steps (recommended)

1. Install [Docker](https://www.docker.com/) for your operating system.
2. Clone the repository:
```bash
git clone https://github.com/aialt/PillagerBench.git
cd PillagerBench
```
3. Set-up your API keys:
- Create a file named `api_keys.py` and add your API keys in this way:
```py
openai_api_key = "..."
deepseek_api_key = "..."
openrouter_api_key = "..."
```
- Place this file in the root of the project directory.
4. Build the Docker image:
```bash
docker build -t PillagerBench .
```
5. Install NPM packages for Mineflayer:
```bash
./js_setup_docker.ps1
``` 
6. Launch the Docker container:
```bash
docker compose up -d
docker attach pillagerbench
```
7. Run a benchmark from config
```bash
python main.py -cn benchmark
```

### Local Installation Steps
1. Install dependencies:
   - Python 3.10
   - Node.js 20 (with NPM)
   - Java 17
2. Clone the repository:
   ```bash
   git clone https://github.com/aialt/PillagerBench.git
   ```
3. Set-up your API keys:
   - Create a file named `api_keys.py` and add your API keys in this way:
   ```py
   openai_api_key = "..."
   deepseek_api_key = "..."
   openrouter_api_key = "..."
   ```
   - Place this file in the root of the project directory.
4. Install NPM packages for Mineflayer:
   ```bash
   ./js_setup.sh
   ```
5. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, try venv\Scripts\activate
   ```
6. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
7. Run a benchmark from config
   ```bash
   python main.py -cn benchmark
   ```

## QuickStart

- Set-up [Hydra](https://hydra.cc/docs/intro/) test configs in the `configs` folder.
- Run your test config:
```bash
python main.py -cn config_name
```
- Observe your test by joining the internal Minecraft server (requires Minecraft 1.19.4). The default address is `localhost:49172`, but the port is configurable.
- Visualize results with `collate_results.py`: (requires editing the file to set options)
```bash
python collate_results.py
```
- Add additional test scenarios by adding classes to the `scenarios` folder that inherit from the `Scenario` base class. You can also add additional world saves to the `bench/mc_server` folder.
- Add additional multi-agent systems by adding classes to the `agents` folder that inherit from the `Agent` base class.

## Credits

This project is the Master's thesis of [Olivier Schipper](https://github.com/OliBomby), check out his other amazing projects!

## Citation

If you find our work helpful, please leave us a star and cite our paper.

```
@INPROCEEDINGS{schipper2025pillagerbench,
  author={Schipper, Olivier and Zhang, Yudi and Du, Yali and Pechenizkiy, Mykola and Fang, Meng},
  booktitle={2025 IEEE Conference on Games (CoG)}, 
  title={PillagerBench: Benchmarking LLM-Based Agents in Competitive Minecraft Team Environments}, 
  year={2025},
  volume={},
  number={},
  pages={1-15},
  keywords={Adaptive learning;Games;Benchmark testing;Solids;Real-time systems;Cognition;Teamwork;Artificial intelligence;Multi-agent systems},
  doi={10.1109/CoG64752.2025.11114387},
  url={https://arxiv.org/abs/2509.06235}
}
```

## License

This project is under the [MIT License](LICENSE).
